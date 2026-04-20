# mpc.py
import casadi as ca
import numpy as np

class CasadiMPCController:
    def __init__(self, horizon=10, dt=0.1, wheelbase=2.5):
        self.H = horizon
        self.dt = dt
        self.L = wheelbase

        # ----- Passive FTC: fixed weights (no online change) -----
        self.weights = {
            'spacing':           50.0,   # gap tracking to predecessor
            'delta_v':           10.0,   # relative velocity regulation
            'leader_velocity':   10.0,   # sync with leader speed
            'leader_position':    0.5,   # formation anchoring to leader pos
            'acceleration':      30.0,   # effort / comfort
            'jerk':               0.8,   # smoothness
            'gap_slack':        500.0,   # soft min-gap penalty
            'ttc':               12.0    # TTC soft-hinge penalty
        }

        # ----- Physical bounds / policy -----
        self.max_accel = 3.5      # |a| <= 3.5 m/s^2
        self.v_min     = 0.0
        self.v_max     = 35.0
        self.d_min     = 5.0      # absolute min spacing (softened by slack)

        # ----- TTC soft-hinge parameters (fixed, passive) -----
        self.T_min  = 2.5         # desired minimal TTC [s]
        self.kappa  = 0.2         # softness for hinge
        self.eps_v  = 1e-2        # to avoid division by ~0 (closing speed)
        self.eps_ttc= 1e-3        # TTC clamp to keep finite

        # If you later want CTH: d0 + h * v_i, you can pass desired_spacing in solve()

    def solve(self, current_state, lead_state, leader_state,
              desired_spacing=10.0, last_u=0.0):
        """
        MPC with fixed weights (passive FTC). Uses:
          - predecessor state (x_lead, v_lead) for spacing/Δv terms,
          - leader state (x_leader, v_leader) for formation anchoring.
        Both states should be the best available (e.g., KF estimates when comm-impaired).
        """
        H, dt = self.H, self.dt
        W = self.weights

        # unpack states (x,y,theta,v) but we only use x,v in longitudinal model
        x0, _, _, v0              = current_state
        x_lead, _, _, v_lead      = lead_state
        x_leader, _, _, v_leader  = leader_state

        opti = ca.Opti()
        x = opti.variable(H + 1)          # follower position predictions
        v = opti.variable(H + 1)          # follower velocity predictions
        a = opti.variable(H)              # control (accel)
        s_gap = opti.variable(H)          # slack (>=0) for soft min-gap

        # initial condition
        opti.subject_to(x[0] == x0)
        opti.subject_to(v[0] == v0)

        cost = 0.0

        # previous input for jerk penalty (parameter to avoid algebraic loop)
        a_prev = opti.parameter()
        opti.set_value(a_prev, float(last_u))  # caller may set to last applied accel

        for t in range(H):
            # ----- Longitudinal kinematics -----
            opti.subject_to(x[t+1] == x[t] + v[t] * dt)
            opti.subject_to(v[t+1] == v[t] + a[t] * dt)

            # ----- Input & state bounds -----
            opti.subject_to(ca.fabs(a[t]) <= self.max_accel)
            opti.subject_to(v[t] >= self.v_min)
            opti.subject_to(v[t] <= self.v_max)

            # ----- Zero-order hold prediction of predecessor and leader -----
            x_lead_t    = x_lead    + v_lead    * dt * t
            v_lead_t    = v_lead
            x_leader_t  = x_leader  + v_leader  * dt * t
            v_leader_t  = v_leader

            # ----- Errors -----
            dx_preceding  = x_lead_t - x[t] - 5           # actual gap to predecessor
            spacing_error = dx_preceding - desired_spacing
            delta_v        = v[t] - v_lead_t               # relative velocity to predecessor

            x_ref_follower = x_leader_t - (5 + desired_spacing)
            dx_leader_ref  = x_ref_follower - x[t]   # leader pos ref (anchoring)
            leader_vel_err = v[t] - v_leader_t

            # ----- Soft min-gap constraint: dx_preceding >= d_min - s_gap[t], s_gap >= 0 -----
            opti.subject_to(s_gap[t] >= 0.0)
            opti.subject_to(dx_preceding + s_gap[t] >= self.d_min)

            # ----- TTC (closing-only) -----
            closing_speed = v[t] - v_lead_t  # >0 means follower is faster (closing)
            ttc_raw = dx_preceding / ca.fmax(closing_speed, self.eps_v)
            TTC = ca.if_else(closing_speed > 0, ttc_raw, 1e6)              # large when not closing
            # soft-hinge penalty: phi = log(1 + exp((T_min - TTC)/kappa))
            phi_ttc = ca.log(1 + ca.exp((self.T_min - ca.fmax(TTC, self.eps_ttc)) / self.kappa))

            # ----- Stage cost (all fixed weights; passive FTC) -----
            cost += W['spacing']         * spacing_error**2
            cost += W['delta_v']         * delta_v**2
            cost += W['leader_velocity'] * leader_vel_err**2
            cost += W['leader_position'] * dx_leader_ref**2
            cost += W['acceleration']    * a[t]**2
            cost += W['gap_slack']       * s_gap[t]**2
            cost += W['ttc']             * phi_ttc

            # jerk penalty
            if t == 0:
                cost += W['jerk'] * (a[t] - a_prev)**2
            else:
                cost += W['jerk'] * (a[t] - a[t-1])**2

        opti.minimize(cost)

        # ----- Solver settings -----
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            return float(sol.value(a[0])), 0.0
        except RuntimeError:
            # solver failed → safe fallback
            return 0.0, 0.0

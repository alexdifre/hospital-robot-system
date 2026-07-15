import numpy as np

from core.execution.ift_engine import CasADiSensitivityComputer


def test_active_set_rejects_distant_obstacles_despite_ipopt_multiplier_residuals():
    computer = object.__new__(CasADiSensitivityComputer)
    computer.active_constraint_tol = 1e-6

    g_ineq = np.array([-28.068, -50.0, -72.582])
    lam_ineq = np.array([3.24e-7, 2.0e-7, 1.25e-7])

    active = computer._active_inequality_indices(g_ineq, lam_ineq)

    assert active.size == 0


def test_active_set_keeps_constraints_on_the_primal_boundary():
    computer = object.__new__(CasADiSensitivityComputer)
    computer.active_constraint_tol = 1e-6

    g_ineq = np.array([-0.5e-6, -2.0e-6, 0.0])
    lam_ineq = np.zeros(3)

    active = computer._active_inequality_indices(g_ineq, lam_ineq)

    np.testing.assert_array_equal(active, np.array([0, 2]))


def test_kkt_primal_sensitivity_exposes_terminal_state_target_jacobian():
    computer = object.__new__(CasADiSensitivityComputer)
    computer.N = 2
    computer.nx = 6
    computer.nu = 3
    computer.n_w = 30
    computer.n_p = 12
    computer.u0_start = 18
    computer.idx_Q = slice(0, 6)
    computer.idx_R = slice(6, 9)
    computer.idx_z_target = slice(10, 12)

    expected = np.arange(12, dtype=float).reshape(6, 2)
    dw_dp = np.zeros((computer.n_w, computer.n_p))
    xN_start = computer.N * computer.nx
    dw_dp[xN_start : xN_start + computer.nx, computer.idx_z_target] = expected

    computer.cost_sensitivity_fn = lambda _w, _lam, _p: np.zeros(computer.n_p)
    computer._compute_primal_sensitivity_active_set = (
        lambda _w, _lam, _p: dw_dp
    )

    sensitivity = computer.compute_sensitivities(
        np.zeros(computer.n_w),
        np.zeros(1),
        np.zeros(computer.n_p),
    )

    assert sensitivity.success
    np.testing.assert_allclose(sensitivity.dxN_dz_target, expected)

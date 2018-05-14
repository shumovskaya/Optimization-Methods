def function_to_minimize_huber(w, A, AA, robots_num, beacons, robots, x, y, gamma):
    n, m = A.shape
    alpha = w[len(w)- robots_num:]
    f_x = A.dot(w[:m]).T - x * np.cos(alpha[robots-1]) - y * np.sin(alpha[robots-1])
    f_y = A.dot(w[m:2*m]).T + x * np.sin(alpha[robots-1]) - y * np.cos(alpha[robots-1])
    return scipy.special.huber(gamma, np.hstack([f_x, f_y])).sum()


def jacobian_huber(w, A, AA, robots_num, beacons, robots, x, y, gamma):
    n, m = A.shape
    alpha = w[len(w)- robots_num:]
    d_alpha = np.zeros((2 * n, len(alpha)))
    coords_indices = np.arange(n)
    d_alpha[coords_indices, robots - 1] = x * np.sin(alpha[robots - 1]) - y * np.cos(alpha[robots - 1])
    d_alpha[coords_indices + n, robots - 1] = x * np.cos(alpha[robots - 1]) + y * np.sin(alpha[robots - 1])
    J = np.hstack([AA, d_alpha])
    f_x = A.dot(w[:m]).T - x * np.cos(alpha[robots-1]) - y * np.sin(alpha[robots-1])
    f_y = A.dot(w[m:2*m]).T + x * np.sin(alpha[robots-1]) - y * np.cos(alpha[robots-1])
    f = np.hstack([f_x, f_y])
    huber_derivative = np.clip(f, -gamma, gamma).reshape(1, len(f))
    return (huber_derivative.dot(J)).flatten()


def solve_problem_BFGS(beacons, robots, x, y, init, gamma=0.1, method='BFGS', max_iter=None):
    robots_num = max(robots)
    beacons_num = max(beacons)
    n = len(robots)
    Ab = np.zeros((n, beacons_num))
    Ar = np.zeros((n, robots_num))
    coords_indices = np.arange(n)
    Ab[coords_indices, beacons - 1] = 1
    Ar[coords_indices, robots - 1] = -1
    A = np.hstack([Ab, Ar])
    AA = np.zeros((2 * n, 2* A.shape[1]))
    AA[:n, :A.shape[1]] = A
    AA[n:, A.shape[1]:] = A
    if method == 'BFGS':
        res = scipy.optimize.minimize(function_to_minimize2, init, method='BFGS',
                                      args=(A, AA, robots_num, beacons, robots, x, y), options={"maxiter": max_iter})
    elif method == 'BFGS2':
        res = scipy.optimize.minimize(function_to_minimize_huber, init, method='BFGS',
                                      args=(A, AA, robots_num, beacons, robots, x, y, gamma), options={"maxiter": max_iter})
    elif method == 'huber':
        res = scipy.optimize.minimize(function_to_minimize_huber, init, jac=jacobian_huber, method='BFGS', 
                        args=(A, AA, robots_num, beacons, robots, x, y, gamma), options={"maxiter": max_iter})
    else:
        raise Exception('Wrong solving method! Use: "BFGS" or "huber"')
    
    b_coords = (res.x[:beacons_num], res.x[beacons_num + robots_num:2*beacons_num + robots_num])
    r_coords = (res.x[beacons_num:beacons_num + robots_num], 
                res.x[2*beacons_num + robots_num:2*beacons_num + 2*robots_num])
    return b_coords, r_coords, res
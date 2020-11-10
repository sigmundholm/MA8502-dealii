import sympy as sp


def u_func(x, y, eps, pck=sp):
    return pck.cos(pck.pi) * (1 - pck.exp((y - 1) / eps)) / (
            1 - pck.exp(-2 / eps)) + 0.5 * pck.cos(pck.pi * x) \
           * pck.sin(pck.pi * y)


def grad(func):
    x, y = sp.var("x y")
    return sp.diff(func, x), sp.diff(func, y)


def div(func):
    x, y = sp.var("x y")
    return sp.diff(func[0], x) + sp.diff(func[1], y)


def laplace(func):
    x, y = sp.var("x y")
    return sp.diff(sp.diff(func, x), x) + sp.diff(sp.diff(func, y), y)


def equation(x, y, eps, b, u, f):
    """
    This implements the operator L st.
        Lu = -εΔu + b·∇u - f

    All arguments must be sympy expressions, except for eps (a number) and b; a
    two dim vector of numbers.
    """
    u_x, u_y = grad(u)
    return -eps * div([u_x, u_y]) + b[0] * u_x + b[1] * u_y - f


if __name__ == '__main__':
    x, y, eps = sp.var("x y eps")

    u = u_func(x, y, eps)
    print("u =", u)

    f = -eps * div(grad(u)) + 1 * grad(u)[1]
    print()
    print("f =", f)
    print("Check if u solves the equation with f as rhs:")
    print("Lu =", equation(x, y, eps, [0, 1], u, f))

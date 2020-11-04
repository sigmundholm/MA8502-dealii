import sympy as sp


def grad(func):
    x, y = sp.var("x y")
    return sp.diff(func, x), sp.diff(func, y)


def div(func):
    x, y = sp.var("x y")
    return sp.diff(func[0], x) + sp.diff(func[1], y)


def equation(x, y, eps, b, u, f):
    """
    This implements the operator L st.
        Lu = -εΔu + b·∇u - f

    All arguments must be sympy expressions, except for eps (a number) and b; a
    two dim vector of numbers.
    """
    u_x, u_y = grad(u)
    return -eps * div([u_x, u_y]) + b[0] * u_x + b[1] * u_y - f


def f_func(u, b):
    x, y, eps = sp.var("x y eps")
    u_x, u_y = grad(u)
    return -eps * div([u_x, u_y]) + b[0] * u_x + b[1] * u_y


def u_1_func():
    x, y, eps = sp.var("x y eps")
    return sp.cos(sp.pi * x) * (1 - sp.exp((y - 1) / eps)) / (1 - sp.exp(-2 / eps)) \
           + 0.5 * sp.cos(sp.pi * x) * sp.sin(sp.pi * y)


def u_2_func():
    x, y, eps = sp.var("x y eps")
    return x * (1 - sp.exp((y - 1) / eps)) / (1 - sp.exp(-2 / eps))


def testit():
    u = u_1_func()
    x, y, eps = sp.var("x y eps")
    laplace = sp.diff(sp.diff(u, x), x) + sp.diff(sp.diff(u, y), y)
    f = - eps * laplace + sp.diff(u, y)
    print("f-test= ", f)
    return f


if __name__ == '__main__':
    b = [0, 1]
    u_1 = u_1_func()
    f_1 = f_func(u_1, b)
    print("u_1 =", u_1)
    print("f_1 =", f_1)
    print("grad u_1 =", grad(u_1))

    u_2 = u_2_func()
    f_2 = f_func(u_2, b)
    print("\nu_2 =", u_2)
    print("f_2 =", f_2)
    print("grad u_2 =", grad(u_2))

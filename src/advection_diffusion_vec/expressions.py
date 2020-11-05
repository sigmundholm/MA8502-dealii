import sympy as sp


def grad(func):
    x, y = sp.var("x y")
    return sp.diff(func, x), sp.diff(func, y)


def div(func):
    x, y = sp.var("x y")
    return sp.diff(func[0], x) + sp.diff(func[1], y)


def laplace(func):
    x, y = sp.var("x y")
    return sp.diff(sp.diff(func, x), x) + sp.diff(sp.diff(func, y), y)


def dot(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def f_func(u: [], b: []) -> []:
    """
    Calculates f for the equation
        Lu = -Δu + b·∇u = f
    """
    u_1, u_2 = u
    f_1 = -laplace(u_1) + dot(b, grad(u_1))
    f_2 = -laplace(u_2) + dot(b, grad(u_2))
    return [f_1, f_2]


def u_1_func():
    x, y, eps = sp.var("x y eps")
    return -sp.cos(sp.pi * x) * sp.sin(sp.pi * y)


def u_2_func():
    x, y, eps = sp.var("x y eps")
    return sp.sin(sp.pi * x) * sp.cos(sp.pi * y)


def testit():
    u = u_1_func()
    x, y, eps = sp.var("x y eps")
    laplace = sp.diff(sp.diff(u, x), x) + sp.diff(sp.diff(u, y), y)
    f = - eps * laplace + sp.diff(u, y)
    print("f-test= ", f)
    return f


if __name__ == '__main__':
    u_1 = u_1_func()
    u_2 = u_2_func()

    # Use u as the convection field b.
    f_1, f_2 = f_func([u_1, u_2], [u_1, u_2])

    print("u_1 =", u_1)
    print("u_2 =", u_2)
    print("f_1 =", f_1)
    print("f_2 =", f_2)

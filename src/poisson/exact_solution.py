import sympy as sp


def grad(func):
    x, y = sp.var("x y")
    return sp.diff(func, x), sp.diff(func, y)


def div(func):
    x, y = sp.var("x y")
    return sp.diff(func[0], x) + sp.diff(func[1], y)


def f_func(u):
    x, y, eps = sp.var("x y eps")
    return -eps * div(grad(u))


def u_1_func():
    x, y, eps = sp.var("x y eps")
    return sp.cos(sp.pi * x) * (1 - sp.exp((y - 1) / eps)) / (1 - sp.exp(-2 / eps)) \
           + 0.5 * sp.cos(sp.pi * x) * sp.sin(sp.pi * y)


def u_2_func():
    x, y, eps = sp.var("x y eps")
    return x * (1 - sp.exp((y - 1) / eps)) / (1 - sp.exp(-2 / eps))


if __name__ == '__main__':
    u_1 = u_1_func()
    f_1 = f_func(u_1)
    print("u_1 =", u_1)
    print("f_1 =", f_1)
    print("grad u =", grad(u_1))

    u_2 = u_2_func()
    f_2 = f_func(u_2)
    print("\nu_2 =", u_2)
    print("f_2 =", f_2)

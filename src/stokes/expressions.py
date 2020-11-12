import sympy as sp

from utils.expressions import grad, laplace


def get_u():
    x, y = sp.var("x y")
    u1 = -sp.cos(sp.pi * x) * sp.sin(sp.pi * y)
    u2 = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    return u1, u2


def get_p():
    x, y = sp.var("x y")
    return -(sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y)) / 4


def get_f(u, p):
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = - laplace(u1) + p_x
    f2 = -laplace(u2) + p_y
    return f1, f2


if __name__ == '__main__':
    u1, u2 = get_u()
    p = get_p()
    f1, f2 = get_f((u1, u2), p)

    print("u_1 =", u1)
    print("u_2 =", u2)
    print("p =", p)

    print("\ngrad u_1 =", grad(u1))
    print("grad u_2 =", grad(u2))
    print("grad p =", grad(p))

    print("\nf_1 =", f1)
    print("f_2 =", f2)

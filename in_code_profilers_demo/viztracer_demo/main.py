from scipy.optimize import minimize
from viztracer import VizTracer



def main():
    # c.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    func = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    x0 = (2, 0)

    constraints = (
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
    )

    bounds = ((0, None), (0, None))

    result = minimize(func, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    print(f"result:\n{result}")
    print(f"best fit parameters: {result.x}")


if __name__ == "__main__":
    with VizTracer(output_file="optional.json") as tracer:
        main()

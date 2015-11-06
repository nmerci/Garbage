#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>
 
void test_gradient_descent();
void test_stochastic_gradient_descent();
 
int main() 
{
    test_gradient_descent();
    test_stochastic_gradient_descent();
 
	return 0;
}
void one_dim_opt(const std::function<double(const double)>& f, double& x)
{
    // TODO
}
 
std::vector<double> gradient_descent(
        const std::function<double(const std::vector<double>&)>& function,
        const std::vector<double>& initial_x, const double initial_step_size,
        const double tolerance, const int max_iterations, const double delta)
{
    /*
     * x    -- argument
     * f    -- target function
     * f_x  -- value of f at x, i.e. f(x)
     */
 
    std::vector<double> x {initial_x};
    double step_size = initial_step_size;
    double f_x {function(x)};
 
    for(int iteration = 0; iteration < max_iterations; ++iteration)
    {
        // calculate gradient
        std::vector<double> grad_f(x.size());
        for(size_t i = 0; i < x.size(); ++i)
        {
            std::vector<double> delta_x {x};
            delta_x[i] += delta;
            grad_f[i] = (function(delta_x) - f_x) / delta;
        }
 
        // update step_size (using one-dimensional optimization)
        step_size *= 0.9; // this technique could be improved
 
        // update x
        for(size_t i = 0; i < x.size(); ++i)
            x[i] -= step_size * grad_f[i];
 
        // update function value at x
        const double f_x_new {function(x)};
 
        // check for convergence        
        if(std::abs(f_x_new - f_x) < tolerance)
        {
            std::cout << "Number of iterations to convergence: "
                      << iteration
                      << std::endl;
            std::cout << "Function value: "
                      << f_x_new
                      << std::endl;
            return x;
        }
        else
            f_x = f_x_new;
    }
 
    std::cerr << "gradient_descent: max number of iterations are exceeded!"
              << std::endl;
    return x;
}
 
void test_gradient_descent()
{
    std::function<double(const std::vector<double>&)> f =
            [](const std::vector<double>& x) -> double
    {
        double sum = 0;
        std::for_each(x.cbegin(), x.cend(), [&](const double i){sum += i*i;});
        return sum;
    };
 
    const std::vector<double> initial_x {1.0, 2.0, 3.0};
    const double step_size {1.0};
    const double tolerance {1e-10};
    const int max_iterations {10000};
    const double delta {0.001};
 
    std::cout << "Test for simple gradient descent" << std::endl;
    std::vector<double> result {gradient_descent(f, initial_x, step_size, tolerance, max_iterations, delta)};
 
    std::cout << "These values have to be close to zero: ";
    for(auto i:result)
        std::cout << i << " ";
    std::cout << std::endl;
}
 
std::vector<double> stochastic_gradient_descent(
        const std::vector<std::function<double(const std::vector<double>&)>>& functions,
        const std::vector<double>& initial_x, const double initial_step_size,
        const double tolerance, const int max_iterations, const double delta)
{
    /*
     * x    -- argument
     * f    -- target function of the form: f = sum(f_i), i=1,...,n
     * fs   -- vector of f_i, i=1,...,n
     * f_x  -- value of f at x, i.e. f(x)
     * fi_x -- value of f_i at x, i.e. f_i(x)
     * fs_x -- vector of values (f_1(x), f_2(x), ..., f_n(x))
     */
 
    std::vector<double> x {initial_x};
    double step_size = initial_step_size;
 
    std::vector<double> fs_x(functions.size());
    for(size_t i = 0; i < functions.size(); ++i)
        fs_x[i] = functions[i](x);
 
    double f_x {std::accumulate(fs_x.cbegin(), fs_x.cend(), 0.0)};
 
    // construct an array of indices 0, 1, 2, ...
    std::vector<size_t> indices(functions.size());
    std::iota(indices.begin(), indices.end(), 0);
 
    for(size_t iteration = 0; iteration < max_iterations / functions.size(); ++iteration)
    {
        for(auto i_function:indices)
        {
            // calculate gradient for specific function
            std::vector<double> grad_f_i(x.size());
            for(size_t i = 0; i < x.size(); ++i)
            {
                std::vector<double> delta_x {x};
                delta_x[i] += delta;
                grad_f_i[i] = (functions[i_function](delta_x) - fs_x[i_function]) / delta;
            }
 
            // update step_size (using one-dimensional optimization)
            step_size *= 0.9; // this technique could be improved
 
            // update x
            for(size_t i = 0; i < x.size(); ++i)
                x[i] -= step_size * grad_f_i[i];
 
            // update function values at x
            for(size_t i = 0; i < functions.size(); ++i)
                fs_x[i] = functions[i](x);
        }
 
        // check for convergence
        const double f_x_new {std::accumulate(fs_x.cbegin(), fs_x.cend(), 0.0)};
        if(std::abs(f_x - f_x_new) < tolerance)
        {
            std::cout << "Number of iterations to convergence: "
                      << iteration * functions.size()
                      << std::endl;
            std::cout << "Function value: "
                      << f_x_new
                      << std::endl;
            return x;
        }
        else
        {
            std::shuffle(indices.begin(), indices.end(),
                         std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
            f_x = f_x_new;
        }
    }
 
    std::cerr << "gradient_descent: max number of iterations are exceeded!"
              << std::endl;
    return x;
}
 
void test_stochastic_gradient_descent()
{
    const size_t n_functions = 3;
    std::vector<std::function<double(const std::vector<double>&)>> functions(n_functions);
    for(size_t i = 0; i < n_functions; ++i)
        functions[i] = [=](const std::vector<double>& x) -> double
                       {
                           return std::pow(x[i], 2);
                       };
 
    const std::vector<double> initial_x {1.0, 2.0, 3.0};
    const double step_size {1.0};
    const double tolerance {1e-10};
    const int max_iterations {10000};
    const double delta {0.001};
 
    std::cout << "Test for stochastic gradient descent" << std::endl;
    std::vector<double> result {stochastic_gradient_descent(functions, initial_x, step_size, tolerance, max_iterations, delta)};
 
    std::cout << "These values have to be close to zero: ";
    for(auto& i:result)
        std::cout << i << " ";
    std::cout << std::endl;
}

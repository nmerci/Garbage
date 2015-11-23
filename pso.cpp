double get_uniform_random_value()
{
    return double(rand()) /  RAND_MAX;
}

vector<double> get_uniform_random_vector(const size_t size)
{
    vector<double> eps(size);
    for_each(eps.begin(), eps.end(), [](double& val) {val = get_uniform_random_value();});
    return eps;
}

std::vector<double> get_uniform_random_vector(const std::pair<std::vector<double>, std::vector<double>>& boundaries)
{
    std::vector<double> random_vector {get_uniform_random_vector(boundaries.first.size())};
    for(size_t i = 0; i < boundaries.first.size() && i < boundaries.second.size(); ++i)
        random_vector[i] = random_vector[i] * (boundaries.second[i] - boundaries.first[i]) + boundaries.first[i];

    return random_vector;
}

std::vector<double> operator^(const double a, const std::vector<double>& v)
{
    std::vector<double> result {v};
    std::for_each(result.begin(), result.end(), [=](double& x) {x = std::pow(a, x);});

    return result;
}

std::vector<double> optimizePSO(const std::function<double(const std::vector<double>&)>& f)
{
    /*
     * All parameters are currently implemented as a private fields
     */
    
    // set up random engine (C++ <random> is not working :/ )
    srand(time(NULL));

    // INITIALIZATION
    // turn boundaries to logarithmic (with base=2) scale
    std::pair<std::vector<double>, std::vector<double>> log2_boundaries {boundaries_};
    std::for_each(log2_boundaries.first.begin(), log2_boundaries.first.end(), [](double& x) {x = std::log2(x);});
    std::for_each(log2_boundaries.second.begin(), log2_boundaries.second.end(), [](double& x) {x = std::log2(x);});

    // initialize particles with random search and its values
    std::vector<std::vector<double>> particles;
    particles.reserve(n_particles_);
    std::vector<double> particle_values;
    particle_values.reserve(n_particles_);

    for(size_t i = 0; i < n_particles_; ++i)
        while(true)
        {
            std::vector<double> x {2^get_uniform_random_vector(log2_boundaries)};
            double x_value {f(x)};

            if(x_value < 1e6) // TODO: remove this kludge
            {
                particles.push_back(x);
                particle_values.push_back(x_value);
                break;
            }
        }

    // initialize global particle number
    size_t global_particle_number {rand() % n_particles_};

    // initialize x with particles
    std::vector<std::vector<double>> x {particles};

    // initialize velocity with zero
    std::vector<std::vector<double>> velocity (n_particles_, std::vector<double>(n_dimensions_, 0.0));

    // OPTIMIZATION
    for(size_t i = 0; i < n_iterations_; ++i)
    {
        for(size_t p = 0; p < n_particles_; ++p)
        {
            for(size_t d = 0; d < n_dimensions_; ++d)
            {
                // update velocity
                velocity[p][d] = phi_velocity_ * velocity[p][d] +
                                 phi_local_ * (particles[p][d] - x[p][d]) * get_uniform_random_value() +
                                 phi_global_ * (particles[global_particle_number][d] - x[p][d]) * get_uniform_random_value();
                // update x
                x[p][d] += velocity[p][d];

                // check x for boundary
                if(x[p][d] < boundaries_.first[d])
                    x[p][d] = boundaries_.first[d];
                if(x[p][d] > boundaries_.second[d])
                    x[p][d] = boundaries_.second[d];
            }

            // update particles
            const double x_value {f(x[p])};
            if(x_value < particle_values[p])
            {
                particles[p] = x[p];
                particle_values[p] = x_value;
            }
        }

        // change global particle number
        global_particle_number = rand() % n_particles_;
    }

    global_particle_number = std::min_element(particle_values.begin(), particle_values.end()) - particle_values.begin();
    return particles[global_particle_number];
}

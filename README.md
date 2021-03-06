# I solve the RBC model using value function iteration, projection, and perturbation methods
I use value function iteration and perturbation methods to solve the standard RBC model and I use projection methods to solve the RBC model with a non-negativity constraint for investment.
    <embed src="https://github.com/shanemcmiken/RBC_3ways/blob/main/src/Model.pdf">
        <p>You can see the model <a href="https://github.com/shanemcmiken/RBC_3ways/blob/main/src/Model.pdf">here</a>.</p>
    </embed>
</object>

# Run code

To run code from Julia REPL:
1. Download files 
2. Type the path name into Julia REPL ";cd [pathname]\RBC_3ways" into REPL
3. To load project, type "] activate ." into REPL
4. To load dependencies, type "] instantiate" 
5. To precompile dependencies, type "] precompile"
6. To run code for perturbation method type "include("RBC_Perturbation.jl")"
7. To run code for projection method type "include("RBC_Projection.jl")"
8. To run code for value function iteration method type "include("RBC_VFI.jl")"


# Value Function Iteration
Below are policy functions for capital and hours produced by my code. I solve the model using value function iteration and I descretize the AR(1) technology process using Tauchen's (1986) method.

![VFI_policyfunc](https://user-images.githubusercontent.com/56058438/126243960-8e98c921-9c7e-4c48-b1c7-29a9097f1d8d.png)

# Projection
Below are policy functions for consumption and hours for the RBC model with a non-negativity constraint for investment. I approximate the policy functions using Chebyshev polynomials collocated at the Chebyshev zeros. I use Smolyak's algorithm to reduce the cardinality of the set of zeros used for the approximation. My code for computing Smolyak zeros and evaluating the polynomial are based on codes by Grey Gordon (2010).

![Projection_policyfunc](https://user-images.githubusercontent.com/56058438/126240310-da579a3a-aa9d-44e9-ac15-0151ae8c5cc5.png)

# Perturbation
Below are RBC impulse responses from a technology shock produced by my code. I solve the model using Schmitt-Grohe and Uribe's (2004) method of first order approximation of the model's policy functions.

![impulseresponses](https://user-images.githubusercontent.com/56058438/126239973-2b6e5078-9931-40b9-a111-8353a070448e.png)!


# Author

Shane McMiken (Boston College, 2021)

# License

This project is licensed under the MIT License.

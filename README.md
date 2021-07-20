# Solve RBC model using value function iteration, projection and perturbation methods


# Value Function Iteration
Below are policy functions for capital and hours produced by my code. I solve the model using value function iteration and I descretize the AR(1) technology process using Tauchen's (1986) method.

![VFI_policyfunc](https://user-images.githubusercontent.com/56058438/126243960-8e98c921-9c7e-4c48-b1c7-29a9097f1d8d.png)

# Projection
Below are policy functions for capital and hours for the RBC model with a non-negativity constraint for investment. I approximate the policy functions using Chebyshev polynomials collocated at the Chebyshev zeros. I use Smolyak's algorithm to reduce the cardinality of the set of zeros used for the approximation. My code for computing Smolyak zeros and evaluating the polynomial are based on codes by Grey Gordon (2010).

![Projection_policyfunc](https://user-images.githubusercontent.com/56058438/126240310-da579a3a-aa9d-44e9-ac15-0151ae8c5cc5.png)

# Perturbation
Below are RBC impulse responses from a technology shock produced by my code. I solve the model using Schmitt-Grohe and Uribe's (2004) method of first order approximation of the model's policy functions.

![impulseresponses](https://user-images.githubusercontent.com/56058438/126239973-2b6e5078-9931-40b9-a111-8353a070448e.png)!


# Author

Shane McMiken (Boston College, 2021)

# License

This project is licensed under the MIT License.

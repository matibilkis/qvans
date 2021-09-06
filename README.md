<br />
<p align="center">


  <h3 align="center">|VANs> </h3>

  <p align="center">
    A semi-agnostic ansatz with variable structure for quantum machine learning
    <br />
    <a href="https://arxiv.org/abs/2103.06712"><strong>Check out ArXiv preprint »</strong></a>
    <br />
    <br />
   <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a> -->
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">VANs: an overview</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## VANs: an overview

How would you walk through the architecture hyperspace to find your favorite quantum circuit? How to do so when both intuition and gradients vanish? Just go for our Variable Ansatz (VAns) algorithm!

<img src="figures_readme/fig1.jpeg" alt="Logo">

QML trains a parametrized quantum circuit to solve a given problem, encoded in some cost function. Depending on the circuit, this approach can potentially run into trouble, since trainability issues and quantum hardware noise essentially forbid the cost function to be minimized.

For this, we very much motivate the idea of optimizing both circuit parameters 𝗮𝗻𝗱 circuit structure in a semi-agnostic fashion :robot: :robot:.

This consists on randomly placing blocks of gates in the circuit, and accept or reject those modifications if the cost function is actually lowered or not. Crucially, we prevent the circuit from over-growing by applying some circuit-compression rules in a problem-informed way.

<img src="figures_readme/fig2.png" alt="Logo">

In turn, this mechanism gets the most out of the available quantum resources. For example, in VQE, we find that cost value (energy) is lower than that of the circuits usually employed, if using the same resources.

<img src="figures_readme/fig3.png" alt="Logo">

### Built With

This implementation of VANs |:shoes:> has been written in Python 3, using
* [Cirq](https://quantumai.google/cirq)
* [TensorFlowQuantum](https://www.tensorflow.org/quantum)
* [OpenFermion](https://quantumai.google/openfermion)


<!-- GETTING STARTED -->
## Getting Started

How to use VANs on a local machine?

### Prerequisites

Have Python 3 installed.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/matibilkis/qvans.git
   ```
2. Optional, but highly recommended. Create a virtual environment to avoid conflict with other dependencies
  ```sh
  python3 -m virtualenv {NameOfVirtualEnv}
  ```
  And activate the virtual environment
  ```sh
  source {NameOfVirtualEnv}/bin/activate.sh
  ```
3. Install libraries
   ```sh
   (NameOfVirtualEnv) pip3 install -r requirements.txt
   ```
4. Now you are ready to use VANs!
  ```sh
  (NameOfVirtualEnv) python3 meta_main.py
  ```


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- CONTRIBUTING -->
## Contributing

Don't hesitate in getting in touch to contribute with this project :)

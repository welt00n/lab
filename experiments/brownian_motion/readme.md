# Browninan Motion

Albert Einstein on the years preceding 1905 was studying thermodynamics,  statical mechanics and theory of heat pretty having studied Boltzmann, Maxwell and Biggs and by the time he was working on his Phd thesis he published 4 papers that were awesome, even he realized that even if not to the extent it came to be, here is his 1905:
- March: Photoelectric effect paper
- April: PhD thesis submitted - A New Determination of Molecular Dimensions
- May: Brownian motion paper
- June: Special relativity
- September: Mass–energy paper

Long story short, Einstein showed that the particles perform a random walk through a fluid they're in if their sizes are comparable to the particles of the fluid. (is that so?really? I have to be careful on my assertions).

We can simulate that by integrating the position of a particle through time given that at each time step they are equally likely to go up, down, left, right, forward or backwards. The easiest way to visualize is to work with only one dimension. Einstein should the there is a linear relation between the spread of the Gaussian and time, the Gaussian spreads linearly in time and with that we can get the diffusion rate of the fluid. (of the fluid? is that so?)

The integration can be done with different levels of sophistication, we will consider fully independent steps so that each step is independent of the last steps and they are all fixed. I dont't think Einstein himself modeled computation and that's not what we're aiming for here. (not yet)

I will however explore different distribution parameters and perhaps even other distributions are interesting to be tested since they are going to show us the most interesting fact, the law of great numbers, indepedently of what we choose the pool our sample random kicks to the particle from, if the number of steps is great enough we should see the macroscopic behavior emerging and the small errors canceling out so that the distribution tends to the Gaussian.

A full list of Einstein papers preceding and including 1905:

- Conclusions Drawn from the Phenomena of Capillarity — 1901 — Investigates surface tension and intermolecular forces, attempting to deduce molecular interactions from macroscopic liquid behavior.

- Kinetic Theory of Thermal Equilibrium and of the Second Law of Thermodynamics — 1902 — Develops a statistical mechanics framework for thermal equilibrium and the second law based on molecular motion.

- A Theory of the Foundations of Thermodynamics — 1903 — Proposes a probabilistic foundation for thermodynamics, emphasizing the role of microscopic states.

- On the General Molecular Theory of Heat — 1904 — Studies energy fluctuations in systems described by molecular kinetic theory, reinforcing the statistical interpretation of thermodynamics.

- On a Heuristic Viewpoint Concerning the Production and Transformation of Light — 1905 — Introduces the light quantum hypothesis to explain the photoelectric effect and other radiation phenomena.

- **On the Motion of Small Particles Suspended in Liquids at Rest Required by the Molecular-Kinetic Theory of Heat — 1905 — Provides the theoretical explanation of Brownian motion and a method to experimentally confirm the existence of atoms.**

- On the Electrodynamics of Moving Bodies — 1905 — Establishes the theory of special relativity and redefines space and time.

- Does the Inertia of a Body Depend Upon Its Energy Content? — 1905 — Shows that energy contributes to inertia, leading to the mass–energy relation E=mc2.

# Goal

- See the 
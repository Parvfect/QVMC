


## SR


"""
1. We get the local energy using jvp - 1.5 h
1. First we go pure ground state without SR (ADAM) - 1 h (network stuff, loss grads, any sampling mistakes) - 1.5 h
2. Second we get the SR and psuedoinvert it and go towards the ground state like that - 1 h
3. We create our many sample limit for TVMC and update the state in that fashion - 1.5 h
4. We do very small   
4. We think about the phase tracking and how that works with the fisher
5. We think about the complex network case
We get dlogpsi/dtheta as we collect samples (will have to use the same procedure for the O computation)
5. 
"""
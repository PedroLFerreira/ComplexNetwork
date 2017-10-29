# ComplexNetwork
A simple analysis library for complex networks.

See folder .\examples to understand how to use the code.

Step 1: Import the library.
Step 2: Initialize a network using net = Network().
Step 3: 
        Import a network using net.Load()
        or
        Create a network using one of our models:
            net.ER_Random()
            net.WS_Random()
            net.BA_Random()
        
        Take a look at the source code if you want to modify any parameters to make a model based on these.
Step 4: Use pre-built functions (ex. net.Degree(node)) to analyze the network. For a list of functions, see the source code.
Step 5: Visualize the network using net.DrawNetwork(). If the network is sufficiently small, you should make useForce=True to make the network more visible. Some other parameters can be modified in the function call, or alternatively, inside the source code.
Step 6: Profit.
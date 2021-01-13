Run Process.py first. Pandas and numpy are the only external libraries required. Then, run Main. There are two
hyperparameters: the embedding dimension size and the size of the hidden layer.

Lastly, you can choose for how many
verses to iterate through. The algorithm is quite slow (by design - meant for large data), so it will take hours to run
the entire bible. Reasonable subsets such as 5 or 20 should take a few minutes at most.

NOTE: this program will create two new files - English.xml and Spanish.xml - in the working directory. You can delete
them and it will build them every time, but I recommend not changing them.
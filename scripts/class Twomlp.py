class Twomlp(nn.Module):
	 def __init__(self,input,hidden,output):
        super().__init__()
		self.fc1 = nn.Linear(input,hidden)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden,output)


		
    def forward(self,x):
		x = self.fc1(x)
		x = self.rel(x)
		x = self.fc2(x)
		return x
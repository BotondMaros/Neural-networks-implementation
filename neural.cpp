
#include "mvector.h"
#include "mmatrix.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////
// Set up random number generation

std::random_device rand_dev;
std::mt19937 rnd(rand_dev());


////////////////////////////////////////////////////////////////////////////////
// Some operator overloads to allow arithmetic with MMatrix and MVector.

// MMatrix * MVector
MVector operator*(const MMatrix &m, const MVector &v)
{
	assert(m.Cols() == v.size());

	MVector r(m.Rows());

	for (int i=0; i<m.Rows(); i++)
	{
		for (int j=0; j<m.Cols(); j++)
		{
			r[i]+=m(i,j)*v[j];
		}
	}
	return r;
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix &m, const MVector &v)
{
	assert(m.Rows() == v.size());

	MVector r(m.Cols());

	for (int i=0; i<m.Cols(); i++)
	{
		for (int j=0; j<m.Rows(); j++)
		{
			r[i]+=m(j,i)*v[j];
		}
	}
	return r;
}

// MVector + MVector
MVector operator+(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i=0; i<lhs.size(); i++)
		r[i] += rhs[i];

	return r;
}

// MVector - MVector
MVector operator-(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i=0; i<lhs.size(); i++)
		r[i] -= rhs[i];

	return r;
}

// MMatrix = MVector <outer product> MVector
// M = a <outer product> b
MMatrix OuterProduct(const MVector &a, const MVector &b)
{
	MMatrix m(a.size(), b.size());
	for (int i=0; i<a.size(); i++)
	{
		for (int j=0; j<b.size(); j++)
		{
			m(i,j) = a[i]*b[j];
		}
	}
	return m;
}

// Hadamard product
MVector operator*(const MVector &a, const MVector &b)
{
	assert(a.size() == b.size());

	MVector r(a.size());
	for (int i=0; i<a.size(); i++)
		r[i]=a[i]*b[i];
	return r;
}

// double * MMatrix
MMatrix operator*(double d, const MMatrix &m)
{
	MMatrix r(m);
	for (int i=0; i<m.Rows(); i++)
		for (int j=0; j<m.Cols(); j++)
			r(i,j)*=d;

	return r;
}

// double * MVector
MVector operator*(double d, const MVector &v)
{
	MVector r(v);
	for (int i=0; i<v.size(); i++)
		r[i]*=d;

	return r;
}

// MVector -= MVector
MVector operator-=(MVector &v1, const MVector &v)
{
	assert(v1.size()==v.size());

	MVector r(v1);
	for (int i=0; i<v1.size(); i++)
		v1[i]-=v[i];

	return r;
}

// MMatrix -= MMatrix
MMatrix operator-=(MMatrix &m1, const MMatrix &m2)
{
	assert (m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

	for (int i=0; i<m1.Rows(); i++)
		for (int j=0; j<m1.Cols(); j++)
			m1(i,j)-=m2(i,j);

	return m1;
}

// Output function for MVector
inline std::ostream &operator<<(std::ostream &os, const MVector &rhs)
{
	std::size_t n = rhs.size();
	os << "(";
	for (std::size_t i=0; i<n; i++)
	{
		os << rhs[i];
		if (i!=(n-1)) os << ", ";
	}
	os << ")";
	return os;
}

// Output function for MMatrix
inline std::ostream &operator<<(std::ostream &os, const MMatrix &a)
{
	int c = a.Cols(), r = a.Rows();
	for (int i=0; i<r; i++)
	{
		os<<"(";
		for (int j=0; j<c; j++)
		{
			os.width(10);
			os << a(i,j);
			os << ((j==c-1)?')':',');
		}
		os << "\n";
	}
	return os;
}


////////////////////////////////////////////////////////////////////////////////
// Functions that provide sets of training data

// Generate 16 points of training data in a pattern that can be classify with an arch
void GetTestData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	x = {{0.125,.175}, {0.375,0.3125}, {0.05,0.675}, {0.3,0.025}, {0.15,0.3}, {0.25,0.5}, {0.2,0.95}, {0.15, 0.85},
		 {0.75, 0.5}, {0.95, 0.075}, {0.4875, 0.2}, {0.725,0.25}, {0.9,0.875}, {0.5,0.8}, {0.25,0.75}, {0.5,0.5}};

	y = {{1},{1},{1},{1},{1},{1},{1},{1},
		 {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}};
}

// Generate 1000 points of test data in a checkerboard pattern
void GetCheckerboardData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	for (int i=0; i<1000; i++)
	{
		x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
		double r = sin(x[i][0]*12.5)*sin(x[i][1]*12.5);
		y[i][0] = (r>0)?1:-1;
	}
}


// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	double twopi = 8.0*atan(1.0);
	for (int i=0; i<1000; i++)
	{
		x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
		double xv=x[i][0]-0.5, yv=x[i][1]-0.5;
		double ang = atan2(yv,xv)+twopi;
		double rad = sqrt(xv*xv+yv*yv);

		double r=fmod(ang+rad*20, twopi);
		y[i][0] = (r<0.5*twopi)?1:-1;
	}
}

// Save the the training data in x and y to a new file, with the filename given by "filename"
// Returns true if the file was saved succesfully
bool ExportTrainingData(const std::vector<MVector> &x, const std::vector<MVector> &y,
						std::string filename)
{
	// Check that the training vectors are the same size
	assert(x.size()==y.size());

	// Open a file with the specified name.
	std::ofstream f(filename);

	// Return false, indicating failure, if file did not open
	if (!f)
	{
		return false;
	}

	// Loop over each training datum
	for (unsigned i=0; i<x.size(); i++)
	{
		// Check that the output for this point is a scalar
		assert(y[i].size() == 1);

		// Output components of x[i]
		for (int j=0; j<x[i].size(); j++)
		{
			f << x[i][j] << " ";
		}

		// Output only component of y[i]
		f << y[i][0] << " " << std::endl;
	}
	f.close();

	if (f) return true;
	else return false;
}




////////////////////////////////////////////////////////////////////////////////
// Neural network class

class Network
{
public:

	// Constructor: sets up vectors of MVectors and MMatrices for
	// weights, biases, weighted inputs, activations and errors
	// The parameter nneurons_ is a vector defining the number of neurons at each layer.
	// For example:
	//   Network({2,1}) has two input neurons, no hidden layers, one output neuron
	//
	//   Network({2,3,3,1}) has two input neurons, two hidden layers of three neurons
	//                      each, and one output neuron
	Network(std::vector<unsigned> nneurons_)
	{
		nneurons = nneurons_; // vector containing layers and number of nodes in layer
		nLayers = nneurons.size(); //number of layers
		weights = std::vector<MMatrix>(nLayers);
		biases = std::vector<MVector>(nLayers);
		errors = std::vector<MVector>(nLayers);
		activations = std::vector<MVector>(nLayers);
		inputs = std::vector<MVector>(nLayers);
		// Create activations vector for input layer 0
		//this is just the first layer represented as MVector
		activations[0] = MVector(nneurons[0]);

		// Other vectors initialised for second and subsequent layers
		for (unsigned i=1; i<nLayers; i++)
		{
			weights[i] = MMatrix(nneurons[i], nneurons[i-1]);
			biases[i] = MVector(nneurons[i]);
			inputs[i] = MVector(nneurons[i]);
			errors[i] = MVector(nneurons[i]); //will store error for each node in a layer
			activations[i] = MVector(nneurons[i]);
		}


	}

	// Return the number of input neurons
	unsigned NInputNeurons() const
	{
		return nneurons[0];
	}

	// Return the number of output neurons
	unsigned NOutputNeurons() const
	{
		return nneurons[nLayers-1];
	}

	// Evaluate the network for an input x and return the activations of the output layer
	MVector Evaluate(const MVector &x)
	{
		// Call FeedForward(x) to evaluate the network for an input vector x
		FeedForward(x);

		// Return the activations of the output layer
		return activations[nLayers-1];
	}


	// Implement the training algorithm - Stochastic gradient descent
	bool Train(const std::vector<MVector> x, const std::vector<MVector> y,
			   double initsd, double learningRate, double costThreshold, int maxIterations)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());

		InitialiseWeightsAndBiases(initsd);

		for (int iter=1; iter<=maxIterations; iter++)
		{
			//random point
			int i = rnd()%x.size();

			FeedForward(x[i]);

			BackPropagateError(y[i]);


			UpdateWeightsAndBiases2(learningRate);

			// update cost after 1000 steps
			if ((!(iter%1000)) || iter==maxIterations)
			{

				double totalcost = TotalCost(x,y);
				std::cout<<iter<<" "<<totalcost<<std::endl;
				if (totalcost < costThreshold){

					return true;
				}
			}
			if(iter==maxIterations){
				double totalcost = TotalCost(x,y);
				std::cout<<iter<<" "<<totalcost<<std::endl;
			}
		} 		return false;
	}



	bool ExportOutput(std::string filename)
	{
		// Check that the network has the right number of inputs and outputs
		assert(NInputNeurons()==2 && NOutputNeurons()==1);

		std::ofstream f(filename);

		// Return false, indicating failure, if file did not open
		if (!f)
		{
			return false;
		}

		// generate a matrix of 250x250 output data points
		for (int i=0; i<=250; i++)
		{
			for (int j=0; j<=250; j++)
			{
				MVector out = Evaluate({i/250.0, j/250.0});
				f << out[0] << " ";
			}
			f << std::endl;
		}
		f.close();

		if (f) return true;
		else return false;
	}


	static bool Test();
	static bool MyTest();

private:
	// Return the activation function sigma
	double Sigma(double z)
	{
		return tanh(z);
	}

	// Return the derivative of the activation function
	double SigmaPrime(double z)
	{
	 	double sech = 2/(exp(-z)+exp(z));
    	return sech*sech;
	}


	// Loop over all weights and biases in the network and set each
	// term to a random number normally distributed with mean 0 and
	// standard deviation "initsd"
	void InitialiseWeightsAndBiases(double initsd){
		assert(initsd>=0);

		// normal random mean 0, var 1
		std::normal_distribution<> dist(0, initsd);


		//looping over all layer
		for (int i=1; i<nLayers; i++)
		{
			//j is looping from 0 to number of neurons in layer
			for (int j=0; j<nneurons[i]; j++){
				for (int k=0; k<nneurons[i-1]; k++){
					//k is from 0 to number of neurons in layer l-1
					weights[i](j,k) = dist(rnd);
				}
				biases[i][j] = dist(rnd);
			}
		}
	}

	// Evaluate the feed-forward algorithm, setting weighted inputs and activations
	// at each layer, given an input vector x
	void FeedForward(const MVector &x){

		assert(x.size() == nneurons[0]);

		activations[0] = x;
		for(int i=1; i<nLayers; i++){ //loops over the layers
			inputs[i] = weights[i]*activations[i-1] + biases[i];

			for (int j=0; j<activations[i].size(); j++){
				activations[i][j] = Sigma(inputs[i][j]);
			}
		}
	}

	// Evaluate the back-propagation algorithm, setting errors for each layer
	void BackPropagateError(const MVector &y)
	{
		assert(y.size() == nneurons[nLayers - 1]);

		//error for the last layer
		for (int i=0; i<nneurons[nLayers-1]; i++){
			errors[nLayers-1][i] = SigmaPrime(inputs[nLayers-1][i]);
			errors[nLayers-1][i] = errors[nLayers-1][i] * (activations[nLayers-1][i]-y[i]);
		}

		//error for any other layers
		for (int j=nLayers-2; j>0; j--){ //j is layer number from L-1 to 2
			for (int i=0; i<nneurons[j]; i++){ //i is number of neurons in layer j
				errors[j][i] = SigmaPrime(inputs[j][i]);
			}
			errors[j] = errors[j] * TransposeTimes(weights[j+1],errors[j+1]);
		}

	}
	// Apply one iteration of the stochastic gradient iteration with learning rate eta.
	void UpdateWeightsAndBiases(double eta)
	{
		// Check that the learning rate is positive
		assert(eta>0);

		MMatrix outer;
		for (int i=1; i<nLayers; i++){ //layers
			outer = OuterProduct(errors[i], activations[i-1]); //outer product for matrix update
			for (int j=0; j<errors[i].size(); j++){ //each node
				biases[i][j] = biases[i][j] - eta*errors[i][j];

				for (int k=0; k<weights[i].Cols(); k++){ //each term in weights
					weights[i](j,k) = weights[i](j,k) - eta*outer(j,k);
				}
			}
		}

	}


	// Return the cost function of the network with respect to a single the desired output y

	double Cost(const MVector &y)
	{
		// Check that y has the same number of elements as the network has outputs
		assert(y.size() == nneurons[nLayers-1]);

		double sum, sub;
		for(int i=0; i<y.size(); i++){
			sub = (y[i]-activations[nLayers-1][i]);
			sum += sub*sub;
		}
		return sum/2 ;
	}

	// Return the total cost C for a set of training data x and desired outputs y
	double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		// Check that there are the same number of inputs as outputs
		assert(x.size() == y.size());

		double totalcost;
		for (int i=0; i<x.size(); i++){ //loops over all the data points in x
			FeedForward(x[i]);
			totalcost += Cost(y[i]);
		}
		return totalcost/x.size();
	}

	// Private member data

	std::vector<unsigned> nneurons;
	std::vector<MMatrix> weights;
	std::vector<MVector> biases, errors, activations, inputs;
	unsigned nLayers;

};



////////////////////////////////////////////////////////////////////////////////
// Main function and example use of the Network class

// Create, train and use a neural network to classify the data in


void ClassifyTestData()
{
	// Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
	Network n({2, 3, 3, 1});

	// Get some data to train the network
	std::vector<MVector> x, y;
	GetTestData(x, y);

	bool trainingSucceeded = n.Train(x, y, 0.1, 0.1, 1e-4, 1000000);

	// If training failed, report this
	if (!trainingSucceeded)
	{
		std::cout << "Failed to converge to desired tolerance." << std::endl;
	}

	// Generate some output files for plotting
	ExportTrainingData(x, y, "test_points.txt");
	n.ExportOutput("test_contour.txt");
}



int main()
{
	ClassifyTestData();
	return 0;
}

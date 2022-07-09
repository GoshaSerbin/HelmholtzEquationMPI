#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "mpi.h"

using namespace std;

typedef double T;

const T PI = 3.1415926535897932384;

#define  MAX(a,b) ((a)>(b)) ? (a) : (b)

#define ITER_LIMIT 50


/* r - red; b - black
^		...	...	... ...
|		r	b	r	...
i		b	r	b	...
		r	b	r	...

	(0;0)	j-> ...
	*/


T f(T x, T y, T k) {													//-nabla u +k^2 u = f(x,y)
	return sin(PI * y) * (2 + k * k * (1 - x) * x + PI * PI * (1 - x) * x);
}

T trueSolve(T x, T y) {
	return sin(PI * y) * (1 - x) * x;
}

void fill(T* u, const int N) {

	for (int i = N; i < N * N; ++i) {
		u[i] = 1;								//u0	//u0 -> u1 -> ... 
	}
	for (int i = 0; i < N; ++i) {				//boundary cond
		u[i * N] = 0;							//u(x, 0)= 0
		u[i * N + N - 1] = 0;					//u(x, 1)= 0
		u[i] = 0;								//u(0, y)= 0
		u[(N - 1) * N + i] = 0;					//u(1, y)= 0
	}
}

void sendUtoLocU(T* u, T* locU, const int N, const int n,  const int myid, const int np) {

	int* displs = nullptr; 
	int* sendcounts = nullptr;
	
	if (myid == 0) {
		displs = new int[np];
		sendcounts = new int[np];
		for (int i = 0; i < np - 1; ++i) {
			sendcounts[i] = N * n;
			displs[i] = N * n * i;			
		}
		displs[np - 1] = N * n * (np - 1); sendcounts[np - 1] = N * n + (N % np) * N;
	}

	
	MPI_Scatterv(u, sendcounts, displs, MPI_DOUBLE, locU, N * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] sendcounts;
	delete[] displs;
}

void sendLocUtoU(T* u, T* locU, const int N, const int n, const int myid, const int np) {

	int* displs = nullptr;
	int* recvcounts = nullptr;

	if (myid == 0) {
		displs = new int[np];
		recvcounts = new int[np];
		for (int i = 0; i < np - 1; ++i) {
			recvcounts[i] = N * n;
			displs[i] = N * n * i;
		}
		displs[np - 1] = N * n * (np - 1); recvcounts[np - 1] = N * n + (N % np) * N;
	}


	MPI_Gatherv(locU, N * n, MPI_DOUBLE, u, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] recvcounts;
	delete[] displs;
}

void findDu(T* locU, T* locPrU, const int N, const int n, T &du) {

	T locMaxdu = 0, a = 0;
	for (int i = 1; i < N * n; ++i) {
		if ((a = fabs(locU[i] - locPrU[i])) > locMaxdu) {
			locMaxdu = a;
		}
	}

	MPI_Allreduce(&locMaxdu, &du, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}


//uses MPI_Send, MPI_Recv
T solveHelmholtzSeidel(const T k, const int N, const int n, T eps, T* u) { //uses new x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	int N0 = ((N % np == 0 || myid != np - 1) ? (myid * n) : (N - n));

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);


	T* bottomNodes = nullptr;			
	T* highNodes = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
	}

	T* copyNodes = new T[N / 2];

	MPI_Status st;

	const int dispHigh = (((N0 + n - 1) % 2 == 0) ? (0) : (1));			//for rednodes
	const int dispBottom = ((N0 % 2 == 0) ? (0) : (1));

	int iter = 0;
	double t = -MPI_Wtime();
	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));
		//sending red nodes
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[2 * i + dispBottom];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD);
		}

		if(myid != np-1)
			MPI_Recv(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &st);

		MPI_Barrier(MPI_COMM_WORLD);
		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[(n - 1) * N + 2 * i + dispHigh];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD);
		}

		if (myid != 0)
			MPI_Recv(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &st);

		
		//calculating nodes, dont use sending nodes

		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {											//calculating local bottom line
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}

		if (myid != np - 1) {										//calculation local high line
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		//sending black nodes
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[2 * i + 1 - dispBottom];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD);
		}
		if (myid != np - 1) MPI_Recv(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		if (myid != np - 1){
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[(n - 1) * N + 2 * i + 1 - dispHigh];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD);
		}
		if(myid != 0) MPI_Recv(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &st);

		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {//calculating local bottom line
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {//calculation local high line
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		

		findDu(locU,locPrU,N,n,du);


		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;


	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodes;
	delete[] bottomNodes;
	delete[] highNodes;


	return t;
}

//uses MPI_SendRecv
T solveHelmholtzSeidelSR(const T k, const int N, const int n, T eps, T* u) { //uses new x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	const int N0 = ((N % np == 0 || myid != np - 1) ? (myid * n) : (N - n));

	const int n0 = ((myid == 0) ? (0) : (N / 2));
	const int nn = ((myid == np - 1) ? (0) : (N / 2));

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);

	T* bottomNodes = nullptr;			
	T* highNodes = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
	}

	T* copyNodes = new T[N / 2];
	const int dispHigh = (((N0 + n - 1) % 2 == 0) ? (0) : (1));			//for rednodes
	const int dispBottom = ((N0 % 2 == 0) ? (0) : (1));

	MPI_Status st;

	double t = -MPI_Wtime();
	int iter = 0;
	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));

		//sending red nodes							//SendRecv
			//wave up
		for (int i = 0; i < nn; ++i) {
			copyNodes[i] = locU[(n - 1) * N + 2 * i + dispHigh];
		}
		MPI_Sendrecv(copyNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, bottomNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		//wave down
		for (int i = 0; i < n0; ++i) {
			copyNodes[i] = locU[2 * i + dispBottom];
		}
		MPI_Sendrecv(copyNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, highNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, MPI_COMM_WORLD, &st);


		//calculating black nodes, dont use sending red nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {											//calculating local bottom line
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}

		if (myid != np - 1) {										//calculation local high line
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}



		//sending black nodes

			//wave up
		for (int i = 0; i < nn; ++i) {
			copyNodes[i] = locU[(n - 1) * N + 2 * i + 1 - dispHigh];
		}
		MPI_Sendrecv(copyNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, bottomNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		//wave down
		for (int i = 0; i < n0; ++i) {
			copyNodes[i] = locU[2 * i + 1 - dispBottom];
		}
		MPI_Sendrecv(copyNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, highNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, MPI_COMM_WORLD, &st);



		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {//calculating local bottom line
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {//calculation local high line
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		findDu(locU, locPrU, N, n, du);

		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;

	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodes;
	delete[] bottomNodes;
	delete[] highNodes;

	return t;
}

//uses immidiate non-blocking communications
T solveHelmholtzSeidelI(const T k, const int N, const int n, T eps, T* u) { //uses new x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	int N0 = ((N % np == 0 || myid != np - 1) ? (myid * n) : (N - n));

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);

	T* bottomNodes = nullptr;		
	T* highNodes = nullptr;
	T* copyNodesUp = nullptr;
	T* copyNodesDown = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
		copyNodesDown = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
		copyNodesUp = new T[N / 2];
	}

	const int dispHigh = (((N0 + n - 1) % 2 == 0) ? (0) : (1));			//for rednodes
	const int dispBottom = ((N0 % 2 == 0) ? (0) : (1));

	
	MPI_Request requp[2];
	MPI_Request reqdown[2];
	MPI_Status stup[2];
	MPI_Status stdown[2];
	if (myid != np - 1) {
		MPI_Send_init(copyNodesUp, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &requp[0]);
		MPI_Recv_init(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &requp[1]);
	}

	if (myid != 0) {
		MPI_Send_init(copyNodesDown, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &reqdown[0]);
		MPI_Recv_init(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &reqdown[1]);
	}
	double t = -MPI_Wtime();
	int iter = 0;
	
	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));

		//sending red nodes							//Immediate
		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesUp[i] = locU[(n - 1) * N + 2 * i + dispHigh];
			}

			MPI_Startall(2, requp);
		}
		if (myid != 0) {

			for (int i = 0; i < N / 2; ++i) {
				copyNodesDown[i] = locU[2 * i + dispBottom];
			}
			MPI_Startall(2, reqdown);
		}

		//calculating black nodes, dont use sending red nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {
			MPI_Waitall(2, reqdown, stdown);
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}
		

		if (myid != np - 1) {	
			MPI_Waitall(2,requp, stup);
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		//sending black nodes
		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesUp[i] = locU[(n - 1) * N + 2 * i + 1 - dispHigh];
			}
			MPI_Startall(2, requp);
		}
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesDown[i] = locU[2 * i + 1 - dispBottom];
			}
			MPI_Startall(2, reqdown);
		}



		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {
			MPI_Waitall(2, reqdown, stdown);
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {
			MPI_Waitall(2, requp, stup);
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		findDu(locU, locPrU, N, n, du);

		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;

	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodesUp;
	delete[] copyNodesDown;
	delete[] bottomNodes;
	delete[] highNodes;

	return t;
}



//uses MPI_Send, MPI_Recv
T solveHelmholtzJacoby(const T k, const int N, const int n, T eps, T* u) { //uses old x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	int N0 = (N % np == 0 || myid != np - 1) ? (myid * n) : (N - n);

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);


	T* bottomNodes = nullptr;
	T* highNodes = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
	}

	T* copyNodes = new T[N / 2];

	MPI_Status st;

	const int dispHigh = ((N0 + n - 1) % 2 == 0) ? (0) : (1);			//for rednodes
	const int dispBottom = (N0 % 2 == 0) ? (0) : (1);

	int iter = 0;
	double t = -MPI_Wtime();
	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));
		//sending red nodes
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[2 * i + dispBottom];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD);
		}

		if (myid != np - 1)
			MPI_Recv(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locU[(n - 1) * N + 2 * i + dispHigh];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD);
		}

		if (myid != 0)
			MPI_Recv(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &st);


		//calculating nodes, dont use sending nodes

		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {											//calculating local bottom line
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}

		if (myid != np - 1) {										//calculation local high line
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		//sending black nodes
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locPrU[2 * i + 1 - dispBottom];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD);
		}
		if (myid != np - 1) MPI_Recv(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodes[i] = locPrU[(n - 1) * N + 2 * i + 1 - dispHigh];
			}
			MPI_Send(copyNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD);
		}
		if (myid != 0) MPI_Recv(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &st);

		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locPrU[i * N + j + 1] + locPrU[i * N + j - 1] + locPrU[(i + 1) * N + j] + locPrU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {//calculating local bottom line
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locPrU[j + 1] + locPrU[j - 1] + locPrU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {//calculation local high line
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locPrU[(n - 1) * N + j + 1] + locPrU[(n - 1) * N + j - 1] + highNodes[j / 2] + locPrU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}



		findDu(locU, locPrU, N, n, du);


		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;


	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodes;
	delete[] bottomNodes;
	delete[] highNodes;


	return t;
}

//uses MPI_SendRecv
T solveHelmholtzJacobySR(const T k, const int N, const int n, T eps, T* u) { //uses new x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	int N0 = (N % np == 0 || myid != np - 1) ? (myid * n) : (N - n);

	int n0 = (myid == 0) ? (0) : (N / 2);
	int nn = (myid == np - 1) ? (0) : (N / 2);

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);

	T* bottomNodes = nullptr;
	T* highNodes = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
	}

	T* copyNodes = new T[N / 2];
	const int dispHigh = (((N0 + n - 1) % 2 == 0) ? (0) : (1));			//for rednodes
	const int dispBottom = ((N0 % 2 == 0) ? (0) : (1));

	MPI_Status st;

	double t = -MPI_Wtime();
	int iter = 0;
	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));

		//sending red nodes							//SendRecv
			//wave up
		for (int i = 0; i < nn; ++i) {
			copyNodes[i] = locU[(n - 1) * N + 2 * i + dispHigh];
		}
		MPI_Sendrecv(copyNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, bottomNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		//wave down
		for (int i = 0; i < n0; ++i) {
			copyNodes[i] = locU[2 * i + dispBottom];
		}
		MPI_Sendrecv(copyNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, highNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, MPI_COMM_WORLD, &st);

		//calculating black nodes, dont use sending red nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {											//calculating local bottom line
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}

		if (myid != np - 1) {										//calculation local high line
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}



		//sending black nodes
			//wave up
		for (int i = 0; i < nn; ++i) {
			copyNodes[i] = locPrU[(n - 1) * N + 2 * i + 1 - dispHigh];
		}
		MPI_Sendrecv(copyNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, bottomNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, MPI_COMM_WORLD, &st);
		MPI_Barrier(MPI_COMM_WORLD);

		//wave down
		for (int i = 0; i < n0; ++i) {
			copyNodes[i] = locPrU[2 * i + 1 - dispBottom];
		}
		MPI_Sendrecv(copyNodes, n0, MPI_DOUBLE, (myid - 1 + np) % np, 42, highNodes, nn, MPI_DOUBLE, (myid + 1) % np, 42, MPI_COMM_WORLD, &st);

		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locPrU[i * N + j + 1] + locPrU[i * N + j - 1] + locPrU[(i + 1) * N + j] + locPrU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {//calculating local bottom line
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locPrU[j + 1] + locPrU[j - 1] + locPrU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {//calculation local high line
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locPrU[(n - 1) * N + j + 1] + locPrU[(n - 1) * N + j - 1] + highNodes[j / 2] + locPrU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		findDu(locU, locPrU, N, n, du);

		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;

	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodes;
	delete[] bottomNodes;
	delete[] highNodes;

	return t;
}

//uses immidiate non-blocking communications
T solveHelmholtzJacobyI(const T k, const int N, const int n, T eps, T* u) { //uses new x_i

	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const T h = 1. / (N - 1);					// x,y in [0;1]
	const T coeff = 4 + k * k * h * h;
	T du = 1.;									// ||u_k - u_k-1||

	int N0 = (N % np == 0 || myid != np - 1) ? (myid * n) : (N - n);

	T* locU = new T[N * n];						//local u for 1 processor
	T* locPrU = new T[N * n];					//local previous u_{k-1} for 1 processor

	//sending parts of u to locU
	sendUtoLocU(u, locU, N, n, myid, np);

	T* bottomNodes = nullptr;
	T* highNodes = nullptr;
	T* copyNodesUp = nullptr;
	T* copyNodesDown = nullptr;
	if (myid != 0) {
		bottomNodes = new T[N / 2];
		copyNodesDown = new T[N / 2];
	}
	if (myid != np - 1) {
		highNodes = new T[N / 2];
		copyNodesUp = new T[N / 2];
	}

	const int dispHigh = ((N0 + n - 1) % 2 == 0) ? (0) : (1);			//for rednodes
	const int dispBottom = (N0 % 2 == 0) ? (0) : (1);


	MPI_Request requp[2];
	MPI_Request reqdown[2];
	MPI_Status stup[2];
	MPI_Status stdown[2];
	if (myid != np - 1) {
		MPI_Send_init(copyNodesUp, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &requp[0]);
		MPI_Recv_init(highNodes, N / 2, MPI_DOUBLE, myid + 1, 42, MPI_COMM_WORLD, &requp[1]);
	}

	if (myid != 0) {
		MPI_Send_init(copyNodesDown, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &reqdown[0]);
		MPI_Recv_init(bottomNodes, N / 2, MPI_DOUBLE, myid - 1, 42, MPI_COMM_WORLD, &reqdown[1]);
	}
	double t = -MPI_Wtime();
	int iter = 0;

	for (; du > eps; ++iter) {

		memcpy(locPrU, locU, n * N * sizeof(T));

		//sending red nodes							//Immediate
		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesUp[i] = locU[(n - 1) * N + 2 * i + dispHigh];
			}

			MPI_Startall(2, requp);
		}
		if (myid != 0) {

			for (int i = 0; i < N / 2; ++i) {
				copyNodesDown[i] = locU[2 * i + dispBottom];
			}
			MPI_Startall(2, reqdown);
		}

		//calculating black nodes, dont use sending red nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locU[i * N + j + 1] + locU[i * N + j - 1] + locU[(i + 1) * N + j] + locU[(i - 1) * N + j] + h * h * f(j * h, (N0 + i) * h, k)) / coeff;
			}
		}

		if (myid != 0) {
			MPI_Waitall(2, reqdown, stdown);
			for (int j = 1 + dispBottom; j < N - 1; j += 2) {		//use sending nodes
				locU[j] = (locU[j + 1] + locU[j - 1] + locU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;
			}
		}


		if (myid != np - 1) {
			MPI_Waitall(2, requp, stup);
			for (int j = 1 + dispHigh; j < N - 1; j += 2) {			//use sending nodes
				locU[(n - 1) * N + j] = (locU[(n - 1) * N + j + 1] + locU[(n - 1) * N + j - 1] + highNodes[j / 2] + locU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		//sending black nodes
		if (myid != np - 1) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesUp[i] = locPrU[(n - 1) * N + 2 * i + 1 - dispHigh];
			}
			MPI_Startall(2, requp);
		}
		if (myid != 0) {
			for (int i = 0; i < N / 2; ++i) {
				copyNodesDown[i] = locPrU[2 * i + 1 - dispBottom];
			}
			MPI_Startall(2, reqdown);
		}



		//calculating red nodes, dont use sending black nodes
		for (int i = 1; i < n - 1; ++i) {
			int dispBlack = 1 + (N0 + i) % 2;
			for (int j = 3 - dispBlack; j < N - 1; j += 2) {
				locU[i * N + j] = (locPrU[i * N + j + 1] + locPrU[i * N + j - 1] + locPrU[(i + 1) * N + j] + locPrU[(i - 1) * N + j] + h * h * f(j * h, (i + N0) * h, k)) / coeff;
			}
		}

		if (myid != 0) {
			MPI_Waitall(2, reqdown, stdown);
			for (int j = 2 - dispBottom; j < N - 1; j += 2) {
				locU[j] = (locPrU[j + 1] + locPrU[j - 1] + locPrU[N + j] + bottomNodes[j / 2] + h * h * f(j * h, N0 * h, k)) / coeff;

			}
		}

		if (myid != np - 1) {
			MPI_Waitall(2, requp, stup);
			for (int j = 2 - dispHigh; j < N - 1; j += 2) {
				locU[(n - 1) * N + j] = (locPrU[(n - 1) * N + j + 1] + locPrU[(n - 1) * N + j - 1] + highNodes[j / 2] + locPrU[(n - 2) * N + j] + h * h * f(j * h, (n - 1 + N0) * h, k)) / coeff;
			}
		}

		findDu(locU, locPrU, N, n, du);

		if (iter == ITER_LIMIT) {
			if (myid == 0)
				cout << "\nWarning! iterations > " << iter << ". Algorithm stopped. \n";
			break;
		}
	}

	t += MPI_Wtime();

	if (myid == 0) cout << "iterations:" << iter << endl;

	sendLocUtoU(u, locU, N, n, myid, np);

	delete[] locU;
	delete[] locPrU;
	delete[] copyNodesUp;
	delete[] copyNodesDown;
	delete[] bottomNodes;
	delete[] highNodes;

	return t;
}


T getError(T* u, const int N) {
	T err = 0;
	T a = 0.;
	T h = 1. / (N - 1);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if ((a = fabs(trueSolve(j * h, i * h) - u[i * N + j])) > err) {
				err = a;
			}
		}

	}
	return err;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int np, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	const int N = 5000;			// mesh N x N
	const T h = 1. / (N - 1);
	const T k = 10. / h;
	const T eps = 1e-8;			// ||u_k+1 - u_k||<eps

	T* u = nullptr;

	u = new T[N * N];
	fill(u, N);

	int n = N / np + (N % np) * (myid == np - 1); //N_y for 1 processor

	cout << (-1+5) % 5;
	T t;


	if (myid == 0) cout << "N = " << N << ",  eps = " << eps << ", np = " << np << endl;

	if (myid == 0)	cout << "\n		Seidel\n";
	t = solveHelmholtzSeidel(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)	cout << "\n		SeidelSR\n";
	t = solveHelmholtzSeidelSR(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)	cout << "\n		SeidelI\n";
	t = solveHelmholtzSeidelI(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)	cout << "\n		Jacoby\n";
	t = solveHelmholtzJacoby(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)	cout << "\n		JacobySR\n";
	t = solveHelmholtzJacobySR(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)	cout << "\n		JacobyI\n";
	t = solveHelmholtzJacobyI(k, N, n, eps, u);
	if (myid == 0) {
		cout << "time: " << t << endl;
		cout << "error: " << getError(u, N) << endl;
		fill(u, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	delete[] u;

	MPI_Finalize();
	return 0;
}
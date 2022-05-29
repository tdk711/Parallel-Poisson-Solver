/* 3D Poisson Equation with Mixed Boundary Conditions*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#define m_printf if (wrank == 0) printf

#define  N   (32+2) // Grid Size (Must be power of 2)
static const double   maxeps = 0.1e-7;
int i,j,k;
double eps;
double T[N][N][N];
double Tk[N][N][N];
double Q[N][N][N];

//Dirichlet BC
double T1 = 0.0; //Top face
double T2 = 0.0; //Bottom face
double T3 = 0.0; //Front face
double T4 = 0.0; //Back face
double T5 = 50.0; //Left face

// Neumann BC:
double dT6 = 50.0; //Right face

double L = 10.0; // Lenght of side of Cube


void print_output();
void compile_array();
void relax(int rank, int size, double d);
void resid();
void init(int rank, int size);
void update_BCs(int rank, int size, double d);
void reassign();

int block, startrow, lastrow;
void update(int rank, int size);
void wtime(double *t)
{
    *t = MPI_Wtime();
}

int main(int argc, char **argv)
{
	int it;
	double time_start, time_fin, d;
    it = 1;

    d = L/(N-3);

    // Initialize MPI
    int wrank, wsize;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    MPI_Barrier(MPI_COMM_WORLD); // wait for all process here

    // Split between processors
    block = N/wsize;
    startrow = block*wrank;
    lastrow =  block*(wrank+1) + 1;
    m_printf("Jacobian 3D started");


    // Initialize matrices 
    init(wrank, wsize);
    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (!wrank) {wtime(&time_start); }

    // Main Loop:
    do
    {
        update_BCs(wrank, wsize, d);
		relax(wrank, wsize, d);
		resid();
        reassign();
        update(wrank,wsize);
        it++;
	}while(eps > maxeps);

    MPI_Barrier(MPI_COMM_WORLD);
    compile_array();

    if(wrank == 0){print_output();}

    if(!wrank){
        wtime(&time_fin);
        printf("Time: %gs\t", time_fin - time_start);
        printf("error is eps = %f\t", eps);
    }

    // Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

	return 0;
}

void init(int rank, int size)
{
    // Initializing the 3D Array:
	for(i = startrow; i <= lastrow; i++){
            for(j = 0; j <= N-1; j++){
            for(k = 0; k <= N-1; k++){
                T[i][j][k] = 0.0;
                Q[i][j][k] = 0.0;
            }
        }
    }

    // Set Sources/Sinks:
   // Q[8][8][8] = 0.0;

    // Setting Dirichlet BCs:
    //Top face
    for(int i = startrow+1; i <= lastrow-1; i++){
        for(int j = 1; j <= N-2; j++){
            T[i][j][N-2] = T1;
        }
    }

    //Bottom face
    for(int i = startrow+1; i <= lastrow-1; i++){
        for(int j = 0; j <= N-2; j++){
            T[i][j][1] = T2;
        }
    }

    //Front face
    for(int i = startrow+1; i <= lastrow-1; i++){
        for(int k = 1; k <= N-2; k++){
            T[i][1][k] = T3;
        }
    }

    //Back face
    for(int i = startrow+1; i <= lastrow-1; i++){
        for(int k = 1; k <= N-2; k++){
            T[i][N-2][k] = T4;
        }
    }

    if(rank == 0){
    //Left face
        for(int j = 1; j <= N-2; j++){
            for(int k = 1; k <= N-2; k++){
                T[1][j][k] = T5;
            }
        }
    }

    // Initializing Tk Array:
	for(i = startrow; i <= lastrow; i++){
        for(j = 0; j <= N-1; j++){
            for(k = 0; k <= N-1; k++){
                Tk[i][j][k] = T[i][j][k];
            }
        }
    }

}

void update_BCs(int rank, int size, double d){
    // Right:
    if(rank == size - 1){
        for(int j = 1; j < N-1; j++){
            for(int k = 1; k < N-1; k++){
                T[lastrow][j][k] = T[lastrow-2][j][k] - dT6*(2.0*d); 
            }
        }
    }
}

void relax(int rank, int size, double d)
{
    if(rank == 0){
        for(i = startrow+2; i <= lastrow-1; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }

    else if(rank == size-1){
        for(i = startrow+1; i <= lastrow-1; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }

    else{
        for(i = startrow+1; i <= lastrow-1; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }
}

void resid()
{
    double subdomain_error;
    double e;
    e = 0.0;

	for(i = startrow+1; i <= lastrow-1; i++){
        for(j = 1; j <= N-2; j++){
            for(k = 1; k <= N-2; k++){
                e += pow((Tk[i][j][k] - T[i][j][k]),2);
            }
        }
    }
    subdomain_error = sqrt(e);
    MPI_Allreduce(&subdomain_error, &eps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

}

void print_output()
{
    FILE* fp;
    char* filename = "T.txt";
    fp = fopen(filename, "wb");

    for(int i = 1; i <= N-2; i++){
        for(int j = 1; j <= N-2; j++){
            for(int k = 1; k <= N-2; k++){
                fprintf(fp, "%f ", Tk[i][j][k]);
            }
            fprintf(fp, "\n");
        }
    }

}

void update(int rank, int size)
{
    MPI_Request request[4];
    MPI_Status status[4];
    // Update neighbours
    if (rank){
        MPI_Isend(&T[startrow+1][0][0], N*N, MPI_DOUBLE, rank-1, 1216, MPI_COMM_WORLD, &request[1]); 
        MPI_Irecv(&T[startrow][0][0], N*N, MPI_DOUBLE, rank-1, 1215, MPI_COMM_WORLD, &request[0]);
    }
    if(rank != size-1) {
        MPI_Isend(&T[lastrow-1][0][0], N*N, MPI_DOUBLE, rank+1, 1215, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&T[lastrow][0][0], N*N, MPI_DOUBLE, rank+1, 1216, MPI_COMM_WORLD, &request[3]);
    }

    // Wait for processes
    int no_of_operations = 4, shift = 0; // all processes 
    if(!rank) { // first process
        no_of_operations = 2;
        shift = 2;
    }
    if(rank == size-1) { // last process
        no_of_operations = 2;
    }
    MPI_Waitall(no_of_operations, &request[shift], &status[0]);

}

void reassign(){
	for(i = startrow; i <= lastrow; i++){
        for(j = 0; j <= N-1; j++){
            for(k = 0; k <= N-1; k++){
                T[i][j][k] = Tk[i][j][k]; ;
            }
        }
    }
}

void compile_array(){
    for(j = 1; j <= N-2; j++){
        for(k = 1; k <= N-2; k++){
            T[startrow][j][k] = 0;
        }
    }

    for(j = 1; j <= N-2; j++){
        for(k = 1; k <= N-2; k++){
            T[lastrow][j][k] = 0;
        }
    }
    MPI_Allreduce(&T[0][0][0], &Tk[0][0][0], N*N*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
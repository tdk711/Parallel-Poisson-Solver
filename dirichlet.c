/* 3D Poisson Equation with Pure Dirichlet Boundary Conditions*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define  N   (32+2) // Grid Size (Must be power of 2)
int i,j,k;
double eps;

double T[N][N][N];
double Tk[N][N][N];
double Q[N][N][N];

//Dirichlet Boundary Conditions
double T1 = 6.0; //Top face
double T2 = -6.0; //Bottom face
double T3 = 0.0; //Front face
double T4 = 0.0; //Back face
double T5 = 0.0; //Left face
double T6 = 0.0; //Right face

double L = 10.0; // Length of side of cube

void print_output();
void compile_array();
void relaxation(int rank, int size, double d);
void L2();
void init(int rank, int size);
void update(int rank, int size);
void reassign();

int block, start_index, last_index;

int main(int argc, char **argv)
{
	int iter_count;
	double time_start, time_finish, d;
    iter_count = 1;

    static const double   error_tolerance = 0.1e-7; // Error Tolerance

    d = L/(N-3);

    // Initialize MPI
    int process_rank, cluster_size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Split between processors
    block = N/cluster_size;
    start_index = block*process_rank;
    last_index =  block*(process_rank+1) + 1;
    if(process_rank == 0) {printf("Jacobian 3D started %d", cluster_size);}


    // Initialize matrices 
    init(process_rank, cluster_size);
    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (!process_rank) {time_start = MPI_Wtime(); }

    // Main Loop
    do
    {
		relaxation(process_rank, cluster_size, d);
		L2();
        reassign();
        update(process_rank,cluster_size);
        iter_count++;
	}while(eps > error_tolerance);

    MPI_Barrier(MPI_COMM_WORLD);
    compile_array();

    if(process_rank == 0){print_output();}

    if(!process_rank){
        time_finish = MPI_Wtime();
        printf("Time: %f\n", time_finish - time_start);
        printf("error is eps = %f\n", eps);
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

	return 0;
}

void init(int rank, int size)
{
    // Initializing the 3D Array:
	for(i = start_index; i <= last_index; i++){
            for(j = 0; j <= N-1; j++){
            for(k = 0; k <= N-1; k++){
                T[i][j][k] = 0.0;
                Q[i][j][k] = 0.0;
            }
        }
    }

    // Set Sources/Sinks:
    //Q[16][16][16] = 100;

    // Setting BCs:
    //Top face
    for(int i = start_index+1; i <= last_index-1; i++){
        for(int j = 1; j <= N-2; j++){
            T[i][j][N-2] = T1;
        }
    }

    //Bottom face
    for(int i = start_index+1; i <= last_index-1; i++){
        for(int j = 0; j <= N-2; j++){
            T[i][j][1] = T2;
        }
    }

    //Front face
    for(int i = start_index+1; i <= last_index-1; i++){
        for(int k = 1; k <= N-2; k++){
            T[i][1][k] = T3;
        }
    }

    //Back face
    for(int i = start_index+1; i <= last_index-1; i++){
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
    
    //Right face
    if(rank == size - 1){
        for(int j = 1; j <= N-2; j++){
            for(int k = 1; k <= N-2; k++){
                T[N-2][j][k] = T6;
            }
        }
    }

    // Initializing Tk Array:
	for(i = start_index; i <= last_index; i++){
        for(j = 0; j <= N-1; j++){
            for(k = 0; k <= N-1; k++){
                Tk[i][j][k] = T[i][j][k];
            }
        }
    }

}

void relaxation(int rank, int size, double d)
{
    if(rank == 0){
        for(i = start_index+2; i <= last_index-1; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }

    else if(rank == size-1){
        for(i = start_index+1; i <= last_index-2; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }

    else{
        for(i = start_index+1; i <= last_index-1; i++){
            for(j = 2; j <= N-3; j++){
                for(k = 2; k <= N-3; k++){
                    Tk[i][j][k] = (T[i-1][j][k] + T[i+1][j][k] + T[i][j-1][k] + T[i][j+1][k] + T[i][j][k-1] + T[i][j][k+1] - (d*d)*Q[i][j][k])/6.0;
                }
            }
        }
    }
}

void L2()
{
    double subdomain_error;
    double e;
    e = 0.0;

	for(i = start_index+1; i <= last_index-1; i++){
        for(j = 2; j <= N-3; j++){
            for(k = 2; k <= N-3; k++){
                if (i == 1 || i == N-2) continue;
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
        MPI_Isend(&T[start_index+1][0][0], N*N, MPI_DOUBLE, rank-1, 1216, MPI_COMM_WORLD, &request[1]); 
        MPI_Irecv(&T[start_index][0][0], N*N, MPI_DOUBLE, rank-1, 1215, MPI_COMM_WORLD, &request[0]);
    }
    if(rank != size-1) {
        MPI_Isend(&T[last_index-1][0][0], N*N, MPI_DOUBLE, rank+1, 1215, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&T[last_index][0][0], N*N, MPI_DOUBLE, rank+1, 1216, MPI_COMM_WORLD, &request[3]);
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
	for(i = start_index+1; i <= last_index-1; i++){
        for(j = 1; j <= N-2; j++){
            for(k = 1; k <= N-2; k++){
                T[i][j][k] = Tk[i][j][k]; ;
            }
        }
    }
}

void compile_array(){
    for(j = 1; j <= N-2; j++){
        for(k = 1; k <= N-2; k++){
            T[start_index][j][k] = 0;
        }
    }

    for(j = 1; j <= N-2; j++){
        for(k = 1; k <= N-2; k++){
            T[last_index][j][k] = 0;
        }
    }
    MPI_Allreduce(&T[0][0][0], &Tk[0][0][0], N*N*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

//Imprimime matrices
void imprimir(const double *matriz, int filas, int columnas) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%.2f ", matriz[i * columnas + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    //Inicialización MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (!rank) fprintf(stderr, "Uso: %s <tamaño de la matriz cuadrada>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }
    
    int n = atoi(argv[1]);
    int n2 = n * n;
    int block_size = n / 2;
    
    //Validar tamaño de matriz y número de procesos
    if (n % 2 != 0) {
        if (!rank) fprintf(stderr, "El tamaño de la matriz debe ser un número par.\n");
        MPI_Finalize();
        return 0;
    }
    
    if (size != 4) {
        if (!rank) fprintf(stderr, "Este programa debe ejecutarse con 4 procesos MPI.\n");
        MPI_Finalize();
        return 0;
    }
    
    //Medición de tiempos
    double t1, t2, elapsed, max_elapsed;
    
    //Creación de matrices A0 y B0
    double *A0 = NULL, *B0 = NULL, *D0 = NULL;
    if (!rank) {
        A0 = (double*)malloc(n2 * sizeof(double));
        B0 = (double*)malloc(n2 * sizeof(double));
        D0 = (double*)malloc(n2 * sizeof(double));
        
        // Inicialización de matrices A0 y B0 locales al proceso 0
        for (int i = 0; i < n2; i++) {
            A0[i] = (double)(rand() % 100) / 10.0;
            B0[i] = (double)(rand() % 100) / 10.0;
        }
        
        // Multiplicación secuencial para posterior verificación de resultados
        t1 = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                D0[i * n + j] = 0.0;
                for (int k = 0; k < n; k++) {
                    D0[i * n + j] += A0[i * n + k] * B0[k * n + j];
                }
            }
        }
        t2 = MPI_Wtime();
        printf("Tiempo secuencial: %f\n", t2 - t1);
    }
    
    //Creación del grid 2D
    MPI_Comm grid, commrow, commcol;
    int ndims = 2;
    int dims[2] = {2, 2};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &grid);
    
    int coords2D[2];
    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, ndims, coords2D);
    int belongs[2];
    
    // Crear subcomunicadores para filas
    belongs[0] = 0; belongs[1] = 1;
    MPI_Cart_sub(grid, belongs, &commrow);
    
    // Crear subcomunicadores para columnas
    belongs[0] = 1; belongs[1] = 0;
    MPI_Cart_sub(grid, belongs, &commcol);
    
    //Creación tipo de datos MPI para facilitar la distribución de las matrices
    MPI_Datatype submatriz_tipo;
    MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &submatriz_tipo);
    MPI_Type_commit(&submatriz_tipo);
    
    //Reserva de memoria de las submatrices
    double *A = (double*)malloc(block_size * block_size * sizeof(double));
    double *B = (double*)malloc(block_size * block_size * sizeof(double));
    double *C = (double*)calloc(block_size * block_size, sizeof(double));
    
    //Distribución de las matrices A y B entre los procesos del grid
    if (!rank) {
        //Llenar submatrices del proceso root
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                A[i * block_size + j] = A0[i * n + j];
                B[i * block_size + j] = B0[i * n + j];
            }
        }
        //Llenar submatrices del resto de procesos mediante el tipo de datos creado
        for (int i = 1; i < size; i++) {
            int coords[2];
            MPI_Cart_coords(grid, i, 2, coords);
            int offset = coords[0] * block_size * n + coords[1] * block_size;
            MPI_Send(&(A0[offset]), 1, submatriz_tipo, i, 0, MPI_COMM_WORLD);
            MPI_Send(&(B0[offset]), 1, submatriz_tipo, i, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(A, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
                           
    double *temp_A = (double*)malloc(block_size * block_size * sizeof(double));
    double *temp_B = (double*)malloc(block_size * block_size * sizeof(double));
    double *temp_C = (double*)calloc(block_size * block_size, sizeof(double));
    
    t1 = MPI_Wtime();
    for (int i = 0; i < 2; i++) {
        // Identificar proceso root para filas y columnas
        int root_row = i;
        int root_col = i;
        
        if (coords2D[0] == root_row) memcpy(temp_B, B, block_size * block_size * sizeof(double));
        if (coords2D[1] == root_col) memcpy(temp_A, A, block_size * block_size * sizeof(double));
        
        //Difusión de filas/columnas necesarias
        MPI_Bcast(temp_A, block_size * block_size, MPI_DOUBLE, root_col, commrow);
        MPI_Bcast(temp_B, block_size * block_size, MPI_DOUBLE, root_row, commcol);
        
        //Cálculo del prodcuto parcial
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                for (int l = 0; l < block_size; l++) {
                    temp_C[j * block_size + k] += temp_A[j * block_size + l] * temp_B[l * block_size + k];
                }
            }
        }
        memcpy(C, temp_C, block_size * block_size * sizeof(double));
    }
    t2 = MPI_Wtime();
    
    free(temp_A); free(temp_B); free(temp_C);
    
    double *C0 = NULL;
    if (!rank) {
        C0 = (double*)malloc(n2 * sizeof(double));
    }
    
    // Recolectar los bloques de la matriz C de todos los procesos
    MPI_Gather(C, block_size * block_size, MPI_DOUBLE, C0, block_size * block_size, MPI_DOUBLE, 0, grid);
    
    //Ordenar matriz C
    if (!rank) {
        double *C_temp = (double*)malloc(n2 * sizeof(double));

        // Reorganizar las submatrices en la matriz temporal C_temp
        for (int proc = 0; proc < size; proc++) {
            int proc_row = (proc / 2) * block_size;
            int proc_col = (proc % 2) * block_size;

            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    int idx_C0 = proc * block_size * block_size + i * block_size + j;
                    int idx_C_temp = (proc_row + i) * n + (proc_col + j);
                    C_temp[idx_C_temp] = C0[idx_C0];
                }
            }
        }

        // Copiar C_temp de vuelta a C0
        memcpy(C0, C_temp, n2 * sizeof(double));
        free(C_temp);
    }
    
    //if (!rank) imprimir(C0, n, n);
    
    elapsed = t2 - t1;
    // Realizar una reducción para encontrar el tiempo máximo
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    //Verificación del resultado es correcto
    if (!rank) {
        double error = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                error += fabs(C0[i * n + j] - D0[i * n + j]);
            }
        }
        printf("Error = %8.2e\nTiempo: %f segundos\n", error, max_elapsed);
    }

    if (!rank) {
        free(A0); free(B0); free(D0); free(C0);
    }
    free(A); free(B); free(C);
    
    MPI_Barrier(grid);
    
    // Liberar comunicadores
    MPI_Comm_free(&grid);
    MPI_Comm_free(&commrow);
    MPI_Comm_free(&commcol);
    
    MPI_Finalize();
    
    return 0;
}

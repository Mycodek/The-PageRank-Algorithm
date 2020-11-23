#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mapreduce.h"
#include "keyvalue.h"
#include "vector"
#include <fstream>
// #include<iostream>
// #include "sys/stat.h"

using namespace MAPREDUCE_NS;
using namespace std;
struct PR_MAT {            // PR_MAT params
    int n;
    vector<vector<int> > H;
    double* I;
    double* A;
};
int check = 0;

// convert_graph_MPI is to convert over matric H to a double variable array so that can be passed to other process
// and convert_MPI_graph is the reverse work for the same.
template <typename T>
double* convert_graph_MPI(std::vector<std::vector<T> > links,int& size)
{
    size = 1;
    for(int i=0;i<links.size();i++)
    {
        size += links[i].size() + 1;
    }   

    double* buff = (double*)malloc(size*sizeof(double));
    int count = 1;
    buff[0] = size-1;
    for(int i=0;i<links.size();i++)
    {
        buff[count] = links[i].size();
        count++;
        for(int j=0;j<links[i].size();j++,count++)
            buff[count] = links[i][j];
    }   
    return buff;
}
template <typename T>
std::vector<std::vector<T> > convert_MPI_graph(double* buff)
{
    std::vector<std::vector<T> > ans;
    int size = buff[0],count=1;
    std::vector<T> temp;
    for(int i=0;count<=size;i++)
    {
        int l = buff[count];
        count++;
        temp.clear();
        for(int j=0;j<l;j++,count++)
            temp.push_back(buff[count]);
        ans.push_back(temp);
    }
    return ans;
}
// function declaration for map,reduce and scan method respectively
void mymap(int, KeyValue *, void *);
void myreduce(char *, int, char *, int, int *, KeyValue *, void *);
void myscan(char *, int, char *, int, void *);

int main(int narg, char **args)
{
    MPI_Init(&narg,&args);
    int me,nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    ifstream fin;
    PR_MAT prmat;
    int nonzeros = 0;
    // ////////////////////////////////////////////////   Preprocessing / Initializing //////////////////////////////////////////
    if(me == 0){
        // looking into input file to get size of required pagerank matrix.
        fin.open("../pagerank-master/test/barabasi-20000.txt");
        int fr,to,max=0;
        string s;
        while(fin){
            fin >> fr >> to;
            max = std::max(max,to);
            max = std::max(max,fr);
        }
        fin.close();
        // found and stored in max # pages on network.
        max++;
        // initializing transpose of graph matrix for only non-zero entry.
        for (int i = 0; i < max; i++)
        {
            vector<int> temp;
            prmat.H.push_back(temp);
        }
        int n = max;
        prmat.n = max;
        fin.open("../pagerank-master/test/barabasi-20000.txt");
        int c = 0;
        while(fin){
            fin >> fr >> to;
            prmat.H[fr].push_back(to);
            c++;
        }
        fin.close();
        // initializing A matric corresponds to dangling nodes (only stored single value per column of H matrix)
        prmat.A = (double *)malloc(n*sizeof(double));
        c = 0;
        for(int i=0;i<n;i++){ 
            if(prmat.H[i].size()==0)
            {
                c = i;
                prmat.A[i] = 1.0/max;
            }
            else
                prmat.A[i] = 0.0;
        }
        // initializing I0 (eign vector)
        prmat.I = (double *)malloc(n*sizeof(double));
        for(int i=0;i<max;i++)
            prmat.I[i] = 0.0;
        prmat.I[0] = 1.0;
        // sending intialized data to every other process.
        int *size_buff = (int*)malloc(sizeof(int));
        int siz;
        double* buff = convert_graph_MPI<int>(prmat.H,siz);
        *size_buff = siz;
        for(int i=1;i<nprocs;i++)
            MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        for(int i=1;i<nprocs;i++)
            MPI_Send(buff, siz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }   
    else
    {
        MPI_Status status;
        int *size_buff = (int*)malloc(sizeof(int));
        MPI_Recv(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int siz = *size_buff;
        double *buff = (double*)malloc(siz*sizeof(double)); 
        MPI_Recv(buff, siz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        prmat.H = convert_MPI_graph<int>(buff);
        prmat.n = prmat.H.size();
        prmat.I = (double *)malloc(prmat.n*sizeof(double));
    }
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////
    MapReduce *mr = new MapReduce(MPI_COMM_WORLD);
    mr->verbosity = 0;
    mr->timer = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();
    int niterate = 0;
    int n = prmat.n;
    while (true){
        // MPI for only changed eign vector
        if(me==0)
        {
            double* buff = (double*)malloc(n*sizeof(double));
            for(int i=0;i<n;i++)
                buff[i] = prmat.I[i];
            for(int i=1;i<nprocs;i++)
                MPI_Send(buff, n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);  
        }
        else
        {
            MPI_Status status;
            double *buff = (double*)malloc(n*sizeof(double)); 
            MPI_Recv(buff, n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            for(int i=0;i<n;i++)
                prmat.I[i] = buff[i];
        }
        
        niterate++;
        // will create key value pair for each two multiplicable number as row as key and I[i] as value. produces KV
        mr->map(prmat.n,&mymap,&prmat);
        // collect all keys from all processes and redistribute according to their keys and produce KMV
        mr->collate(NULL);
        // work on KMV and produce KV according to myreduce function call
        mr->reduce(&myreduce,&prmat);
        // gather all KV from every process to process 0.
        mr->gather(1);

        // initialized to store computed Ik column vector.
        double *HIk = (double *)malloc(prmat.n*sizeof(double));
        for (int i = 0; i < prmat.n; i++){
            HIk[i] = 0.0;
        }
        // scan data from all process and store it in HIk, only process zero will work on this as other does not have any data.
        mr->scan(&myscan,HIk);
        // pid 0 will compute Ik+1 and check for convergence others will only respond to pid 0 wehter to break loop or get udated Ik and do again.
        if(me == 0){
            double alpha = 0.85;
            double add=0.0; //add = alpha*AIk + ((1-alpha)/n)Ik
            for(int i=0;i<prmat.n;i++){
                add += alpha*prmat.A[i]*prmat.I[i] + (1-alpha)*prmat.I[i]/prmat.n; 
            }
            // Ik = alpha * HIk + ( alpha*AIk + ((1-alpha)/n)Ik )
            double err = 0.0;
            bool exit_it = true;
            for (int i = 0; i < prmat.n; i++){
                double oldI = prmat.I[i];
                prmat.I[i] = alpha*HIk[i] + add;
                // error calculation/////////////////////////
                double err_diff  = std::abs(prmat.I[i]-oldI);
                err += err_diff;
                if(err_diff >0.00001) exit_it=false;
            }
            if(err<0.000001) 
            {
                int *size_buff = (int*)malloc(sizeof(int));
                *size_buff = 1;
                for(int i=1;i<nprocs;i++)
                    MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                break;
            }
            else 
            {
                int *size_buff = (int*)malloc(sizeof(int));
                *size_buff = 0;
                for(int i=1;i<nprocs;i++)
                    MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // cout<<"error "<<err<<", in iteration number "<<niterate<<endl;
            }
        }
        else    
        {
            MPI_Status status;
            int *size_buff = (int*)malloc(sizeof(int));
            MPI_Recv(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if(*size_buff==1)
                break;
        }
        // cout << me << " " << check << endl;
        check = 0;
    }
    delete mr;
    MPI_Barrier(MPI_COMM_WORLD);
    double tstop = MPI_Wtime();

    if (me == 0) {
        cout<<"total time : "<<tstop-tstart<<endl;
        cout<<"iterations : "<<niterate<<endl;
        fin.open("../pagerank-master/test/barabasi-20000-pr-p.txt");
        std::string s1,s2;
        double f,fmax=0,tot=0.0;
        for(int i=0;i<prmat.n;i++)
        {
            fin >> s1 >> s2 >> f;
            double temp = std::abs(prmat.I[i]-f);
            tot += temp;
            fmax = std::max(fmax,temp);
        }
        std::cout<<"max error = "<<fmax<<"\nsum error = "<< tot << "\n";

        fin.close();
        // storing final eign vector to output .txt file where ith row rep as : page i = double value of it's importance
        ofstream result_file;
        result_file.open ("output.txt");
        for(int i =0 ;i<prmat.n;i++){
            result_file <<i<<" = "<<prmat.I[i]<<"\n";
        }
        result_file.close();
    }
    MPI_Finalize();
    return 0;
}

/* ----------------------------------------------------------------------
    for each non-zero value of 'itask column' in matrix, emits (key = row_number, value = matrix_value * eign_value)
------------------------------------------------------------------------- */
void mymap(int itask, KeyValue *kv, void *ptr){
    int me;
    check++;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    PR_MAT *prmat = (PR_MAT *) ptr;
    int m = prmat->H[itask].size();
    for (int loop=0; loop<m; loop++){
        int* emit_key = (int *)malloc(sizeof(int));
        double* res_val = (double *)malloc(sizeof(double));
        *emit_key = prmat->H[itask][loop];
        *res_val = (prmat->I[itask])/m;
        // if(*emit_key == 0)
            // check += *res_val;
        kv->add((char *) (emit_key),sizeof(int),(char *) (res_val), sizeof(double));
    }
}

/* ----------------------------------------------------------------------
    for each value corresponding to key in KMV compute their sum. emits (key,sum)
------------------------------------------------------------------------- */
void myreduce(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, KeyValue *kv, void *ptr){
    double* myvalues = (double *) multivalue;
    double sum = 0;
    for(int i=0;i<nvalues;i++){
        sum += myvalues[i];
    }
    double* fsum = (double *)malloc(sizeof(double));
    // double *fsum = malloc();
    *fsum = sum;
    kv->add(key,keybytes,(char *) fsum,sizeof(double));
}

void myscan(char *key, int keybytes, char *value, int valuebytes, void *ptr){
    // cout << "hi3" << endl;
    int k = *((int *)key);
    double val = *((double *)value);
    ((double *)ptr)[k] = val;
}

// to excecute it
// mpic++ -m64 -g -O -I../mrmpi/mrmpi-7Apr14/src  -c part3.cpp
// mpic++ -g -O part3.o ../mrmpi/mrmpi-7Apr14/src/libmrmpi_mpicc.a  -o part3
// mpirun -np 8 part3
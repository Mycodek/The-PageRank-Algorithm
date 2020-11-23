#include "my_mapreduce.h"
#include <fstream>

void My_Mapred::my_map(int itask, KeyValue *kv, void *links_ptr, void *eign_ptr)//type issue
{
	std::vector<std::vector<int> > *links = (std::vector<std::vector<int> >*)links_ptr;
	std::vector<double> *eign = (std::vector<double>*)eign_ptr; 
	int len = links->at(itask).size();
	for (int i=0;i<len;i++)
		kv->add(links->at(itask)[i],eign->at(itask)/len);
}

std::pair<int,double> My_Mapred::my_reduce(KeyMultiValue kmv)
{
	double val = 0;
	for(int i=0;i<kmv.values.size();i++)
		val += kmv.values[i];
	return std::pair<int,double>(kmv.key,val);
}

// void display_kv(KeyValue kv,int me)
// {
// 	for(int i=0;i<kv.get_num_pairs();i++)
// 	{
// 		std::pair<int,double> temp = kv.get_pair(i);
// 		//std::cout << me << " | " << temp.first << " -> " << temp.second << "\n";
// 	}
// }

// void display_link(std::vector<std::vector<int> > links)
// {
// 	for(int i=0;i<links.size();i++)
// 	{
// 		for(int j=0;j<links[i].size();j++)
// 			//std::cout << links[i][j] << " ";
// 		//std::cout << "\n";
// 	}
// }

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

int main()
{
	int comm_sz,my_rank;
	MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    std::vector<std::vector<int> > links; 
	std::vector<double> eign;
	int n;
	double alpha = 0.85;
	std::vector<double> A;
	int *size_buff;
	std::ifstream fin;

	if(my_rank==0)
	{
	    fin.open("./pagerank-master/test/erdos-100000.txt");
	    int fr,to,max=0;
	    while(fin)
	    {
	        fin >> fr >> to;
	        max = std::max(max,to);
	        max = std::max(max,fr);
	    }
	    fin.close();
	    max++;
	    n = max;
	    //std::cout << "number of pages = " << n <<"\n";
	    for(int i=0;i<n;i++)
	    {
	        std::vector<int> temp;
	        links.push_back(temp);
	    }
	    fin.open("./pagerank-master/test/erdos-100000.txt");
	    while(fin)
	    {
	        fin >> fr >> to;
	        links[fr].push_back(to);
	    }
	    fin.close();

		size_buff = (int*)malloc(sizeof(int));
		int siz;
		double* buff = convert_graph_MPI<int>(links,siz);
		*size_buff = siz;
		for(int i=1;i<comm_sz;i++)
			MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		for(int i=1;i<comm_sz;i++)
			MPI_Send(buff, siz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);

	    for(int i=0;i<links.size();i++)
	    { 
	        if(links[i].size()==0)
	            A.push_back(1.0/n);
	        else
	            A.push_back(0);
	    }
	    // for(int i=0;i<links[34].size();i++)
     //        std::cout << links[34][i] << " ";
     //    std::cout << std::endl;

	    for(int i=0;i<n;i++)
	        eign.push_back(1.0/n);
	    //std::cout << "preprocessing done!\n";
	}
	else
	{
		MPI_Status status;
		size_buff = (int*)malloc(sizeof(int));
		MPI_Recv(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		int siz = *size_buff;
		double *buff = (double*)malloc(siz*sizeof(double)); 
		MPI_Recv(buff, siz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
		links = convert_MPI_graph<int>(buff);
		n = links.size();
	}

	int count =0;
    while(true)
    {   
    	if(my_rank==0)
    	{
    		//std::cout << count << " " << eign[0] <<"\n";
    		std::vector<std::vector<double> > eign_t;
    		eign_t.push_back(eign);
    		size_buff = (int*)malloc(sizeof(int));
			int siz;
			double* buff = convert_graph_MPI<double>(eign_t,siz);
			*size_buff = siz;
			for(int i=1;i<comm_sz;i++)
				MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			for(int i=1;i<comm_sz;i++)
				MPI_Send(buff, siz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);	
    	}
    	else
    	{
    		MPI_Status status;
    		size_buff = (int*)malloc(sizeof(int));
			MPI_Recv(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			int siz = *size_buff;
			double *buff = (double*)malloc(siz*sizeof(double)); 
			MPI_Recv(buff, siz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
			eign = convert_MPI_graph<double>(buff)[0];
    	}

    	My_Mapred mr = My_Mapred(MPI_COMM_WORLD);
    	std::vector<std::vector<int> > *links_ptr = &links; 
		std::vector<double> *eign_ptr = &eign;
        mr.map(n,links_ptr,eign_ptr);
	    mr.collate();
	    mr.reduce();
	    mr.gather();

	    if(my_rank==0)
	    {
	    	count++;
        	bool b=1;
		    KeyValue kv = mr.getReducedKV();
	        std::vector<double> temp(eign.begin(),eign.end());
	        for(int i=0;i<n;i++)
	            eign[i] = 0;
	        //alpha*H*I
	        for (int i=0;i<kv.get_num_pairs();i++)
	        {
	            eign[kv.get_pair(i).first]=alpha*(kv.get_pair(i).second);
	        }
	        //(alpha*A+(1-alpha)/n*1)*I
	        double add=0.0;
	        for(int i=0;i<n;i++)
	        {
	            add += alpha*A[i]*temp[i] + (1-alpha)*temp[i]/n; 
	        }
	        // std::cout << add << "\n";
	        // std::cout << eign[0] << " ";
	        for(int i=0;i<n;i++)
	            eign[i] += add;

	        // std::cout << eign[0] << "\n";
	        double err = 0.0;
	        for(int i=0;i<n;i++)
	        {
	            double t = std::abs(eign[i]-temp[i]);
	            err += t;
	            if(t>0.00001)
	                b=0;
	        }
	        if(err<0.000001)
	        {
	        	*size_buff = 1;
				for(int i=1;i<comm_sz;i++)
					MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	            break;
	        }
	        else
	        {
	        	*size_buff = 0;
				for(int i=1;i<comm_sz;i++)
					MPI_Send(size_buff, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	        }
	    }
	    else
	    {
	    	MPI_Status status;
	    	MPI_Recv(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	    	if(*size_buff==1)
	    		break;
	    }
    }

    if(my_rank==0)
    {
    	std::cout << "\n" << count << " iterations\n";
    	fin.open("./pagerank-master/test/erdos-100000-pr-p.txt");
	    std::string s1,s2;
	    double f,fmax=0,tot=0.0;
	    for(int i=0;i<n;i++)
	    {
	        fin >> s1 >> s2 >> f;
	        double temp = std::abs(eign[i]-f);
	        tot += temp;
	        fmax = std::max(fmax,temp);
	        // if(i<20)	
	        	// std::cout << f <<"\n";
	    }
	    std::cout<<"max error = "<<fmax<<"\nsum error = "<< tot << "\n";
	    fin.close();
    }

	MPI_Finalize();
	return 0;
}

// mpiexec -n 1 ./a.out -> 6.233s
// mpiexec -n 2 ./a.out -> 5.722s
// mpiexec -n 3 ./a.out -> 5.600s
// mpiexec -n 4 ./a.out -> 5.267s
// mpiexec -n 5 ./a.out -> 6.664s
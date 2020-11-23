#include<bits/stdc++.h>
#include <mpi.h>

struct KeyMultiValue
{
	int key;
	std::vector<double> values;
};

void print_kmv(KeyMultiValue kmv,int me)
{
	std::cout << me << " | " << kmv.key << " -> ";
	for(int i=0;i<kmv.values.size();i++)
		std::cout << kmv.values[i] << " ";
}

double* convert_KMV_MPI(KeyMultiValue kmv,int& size)
{
	int n = kmv.values.size();
	size = n+2;
	double* buff = (double*)malloc((n+2)*sizeof(double));
	buff[0] = kmv.key;
	buff[1] = n;
	for(int i=0;i<n;i++)
		buff[i+2] = kmv.values[i];
	return buff;
}

KeyMultiValue convert_MPI_KMV(double* buff)
{
	KeyMultiValue kmv;
	kmv.key = (int)buff[0];
	int n = (int)buff[1];
	for(int i=0;i<n;i++)
		kmv.values.push_back(buff[i+2]);
	return kmv;
}

class KeyValue
{
private:
	std::vector<std::pair<int,double>> kv_pairs;
public:
	void add(int key, double value)
	{
		kv_pairs.push_back(std::pair<int,double>(key,value));
	}

	int get_num_pairs()
	{
		return kv_pairs.size();
	}

	std::pair<int,double> get_pair(int i)
	{
		return kv_pairs[i];
	}

	void concat(KeyValue kv)
	{
		this->kv_pairs.insert(this->kv_pairs.end(),kv.kv_pairs.begin(),kv.kv_pairs.end());
	}

	void reset()
	{
		kv_pairs.clear();	
	}
};

double* convert_KV_MPI(KeyValue kv,int& size)
{
	int n = kv.get_num_pairs();
	size = 2*n+1;
	double* buff = (double*)malloc((2*n+1)*sizeof(double));
	buff[0] = n;
	for(int i=0;i<n;i++)
	{
		buff[2*i+1] = kv.get_pair(i).first;
		buff[2*i+2] = kv.get_pair(i).second;
	}
	return buff;
}

KeyValue convert_MPI_KV(double* buff)
{
	KeyValue kv;
	int n = (int)buff[0];
	for(int i=0;i<n;i++)
	{
		kv.add((int)buff[2*i+1],buff[2*i+2]);
	}
	return kv;
}

class My_Mapred
{
private:
	MPI_Comm mpi_comm;
	int me,n_processes;
	std::vector<KeyValue> kv_objs;
	std::vector<KeyMultiValue> kmv_objs;
	KeyValue final_kv;
public:
	My_Mapred(MPI_Comm comm)
	{
		mpi_comm = comm;
		MPI_Comm_size(comm, &n_processes);
    	MPI_Comm_rank(comm, &me);
	}

	KeyValue getReducedKV()
	{
		return final_kv;
	}

	void my_map(int, KeyValue *, void *, void*);
	std::pair<int,double> my_reduce(KeyMultiValue kmv);

	void map(int nmaps, void *ptr1, void *ptr2)
	{
		for(int i=0; i<nmaps/n_processes; i++)
		{
			KeyValue kv = KeyValue();
			my_map(me*(nmaps/n_processes)+i,&kv,ptr1,ptr2);
			kv_objs.push_back(kv);
		}
		if((nmaps/n_processes)*n_processes + me < nmaps)
		{
			KeyValue kv = KeyValue();
			my_map((nmaps/n_processes)*n_processes+me,&kv,ptr1,ptr2);
			kv_objs.push_back(kv);
		}
	}

	void collate()
	{
		if(me==0)
		{
			std::unordered_map<int,KeyMultiValue> hash_table;
			for(int i=0;i<kv_objs.size();i++)
			{
				for(int j=0;j<kv_objs[i].get_num_pairs();j++)
				{
					std::pair<int,double> kvp = kv_objs[i].get_pair(j);
					if(hash_table.find(kvp.first)==hash_table.end())
					{
						KeyMultiValue kmv;
						kmv.key = kvp.first;
						kmv.values.push_back(kvp.second);
						hash_table[kvp.first] = kmv;
					}
					else
						hash_table[kvp.first].values.push_back(kvp.second);
				}
			}

			MPI_Status status;
			int *size_buff = (int*)malloc(sizeof(int));
			int total[n_processes];
			int max_tot = 0;

			for(int i=1;i<n_processes;i++)
			{
				MPI_Recv(size_buff, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				total[status.MPI_SOURCE] = *size_buff;
				max_tot = std::max(*size_buff,max_tot);
			}

			for(int i=1;i<n_processes;i++)
			{
				double *buff = (double*)malloc((2*total[i])*sizeof(double));
				MPI_Recv(buff, 2*total[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
				for(int j=0; j<total[status.MPI_SOURCE]; j++)
				{
					int k = buff[2*j];
					double v = buff[2*j+1];
					if(hash_table.find(k)==hash_table.end())
						{
							KeyMultiValue kmv;
							kmv.key = k;
							kmv.values.push_back(v);
							hash_table[k] = kmv;
						}
					else
						hash_table[k].values.push_back(v);
				}
			}

			std::unordered_map<int,KeyMultiValue>::iterator it;
			int num_keys = hash_table.size();

			*size_buff = num_keys;
			for(int i=1;i<n_processes;i++)
				MPI_Send(size_buff, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
			int count;
			for(it = hash_table.begin(),count=0;it!=hash_table.end();it++,count++)
			{
				int siz,proc=count;
				if((num_keys/n_processes)!=0)
					proc = count/(num_keys/n_processes);
				if(proc==n_processes)
					proc = count - (num_keys/n_processes)*n_processes;
				if(proc!=0)
				{
					double* buff = convert_KMV_MPI(it->second,siz);
					*size_buff = siz;
					MPI_Send(size_buff, 1, MPI_INT, proc, 3, MPI_COMM_WORLD);
					MPI_Send(buff, siz, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD);
				}
				else
				{
					kmv_objs.push_back(it->second);
				}
			}
		}
		else
		{
			int n_maps = kv_objs.size();
			int *size_buff = (int*)malloc(sizeof(int));

			*size_buff = 0;
			for(int i=0;i<n_maps;i++)
				*size_buff += kv_objs[i].get_num_pairs();
			MPI_Send(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

			double *buff = (double*)malloc(2*(*size_buff)*sizeof(double));
			int count = 0;
			for(int i=0;i<n_maps;i++)
			{
				for(int j=0;j<kv_objs[i].get_num_pairs();j++,count+=2)
				{
					std::pair<int,double> kvp = kv_objs[i].get_pair(j);	
					buff[count] = kvp.first;
					buff[count+1] = kvp.second;
				}
			}
			MPI_Send(buff, 2*(*size_buff), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Status status;
			MPI_Recv(size_buff, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
			int num_keys = *size_buff;

			for(int i=0; i<num_keys/n_processes; i++)
			{
				MPI_Recv(size_buff, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
				int siz = *size_buff;
				double* buff = (double*)malloc(siz*sizeof(double));
				MPI_Recv(buff, siz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
				kmv_objs.push_back(convert_MPI_KMV(buff));
			}
			if((num_keys/n_processes)*n_processes + me < num_keys)
			{
				MPI_Recv(size_buff, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
				int siz = *size_buff;
				double* buff = (double*)malloc(siz*sizeof(double));
				MPI_Recv(buff, siz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
				kmv_objs.push_back(convert_MPI_KMV(buff));
			}	
		}
	}

	void reduce()
	{
		final_kv = KeyValue();
		std::pair<int,double> temp;
		for(int i=0;i<kmv_objs.size();i++)
		{
			temp = my_reduce(kmv_objs[i]);
			final_kv.add(temp.first,temp.second);
		}
	}

	void gather()
	{
		if(me==0)
		{
			MPI_Status status;
			int *size_buff = (int*)malloc(sizeof(int));
			int len[n_processes];

			for(int i=1;i<n_processes;i++)
			{
				MPI_Recv(size_buff, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				len[status.MPI_SOURCE] = *size_buff;
			}

			for(int i=1;i<n_processes;i++)
			{
				double *buff = (double*)malloc((len[i])*sizeof(double));
				MPI_Recv(buff, len[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
				this->final_kv.concat(convert_MPI_KV(buff));
			}
		}
		else
		{
			int siz;
			double *buff = convert_KV_MPI(final_kv,siz);
			int *size_buff = (int*)malloc(sizeof(int));
			*size_buff = siz;
			MPI_Send(size_buff, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(buff, siz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			final_kv.reset();
		}
	}
};

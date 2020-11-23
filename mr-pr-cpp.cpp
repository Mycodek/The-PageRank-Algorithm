// Copyright (c) 2009-2016 Craig Henderson
// https://github.com/cdmh/mapreduce
#include <bits/stdc++.h>
#include <boost/config.hpp>
#include <fstream>
#include <ctime>
#if defined(BOOST_MSVC)
#   pragma warning(disable: 4127)

// turn off checked iterators to avoid performance hit
#   if !defined(__SGI_STL_PORT)  &&  !defined(_DEBUG)
#       define _SECURE_SCL 0
#       define _HAS_ITERATOR_DEBUGGING 0
#   endif
#endif

#include"../mapreduce-develop/include/mapreduce.hpp" 

namespace hyperlink {

std::vector<std::vector<int> > links; 
std::vector<double> eign;
int n;
template<typename MapTask>
class datasource : mapreduce::detail::noncopyable
{
  public:
    datasource() : sequence_(0)
    {
    }

    bool const setup_key(typename MapTask::key_type &key)
    {
        key = sequence_++;
        return key < n;
    }

    bool const get_data(typename MapTask::key_type const &key, typename MapTask::value_type &value)
    {
        value = links[key];
        return true;
    }

  private:
    unsigned sequence_;
};

struct map_task : public mapreduce::map_task<unsigned, std::vector<int> >
{
    template<typename Runtime>
    void operator()(Runtime &runtime, key_type const &key, value_type const &value) const
    {
        // std::cout << "\n\n" << eign[key] << "\n";
        // std::cout << key << "\n";
        // if(key==0)
        //     std::cout<<value.size()<<"\n";
        for (auto const &v1 : value)
        {
            typename Runtime::reduce_task_type::key_type const emit_key = v1;
            double res = eign[key]/(value.size());
            // std::cout << emit_key << " " << res << "\n";
            runtime.emit_intermediate(emit_key, res);
        }
        // std::cout<<"yo\n";
    }
};

//called only when rerduction required
struct reduce_task : public mapreduce::reduce_task<unsigned, double >
{
    template<typename Runtime, typename It>
    void operator()(Runtime &runtime, key_type const &key, It it, It ite) const
    {
        if (it == ite)
            return;

        else if (std::distance(it,ite) == 1)
        {
            runtime.emit(key, *it);
            return;
        }

        // calculate the addition of all values in (it .. ite]
        double res = 0;
        for (It it1=it; it1!=ite; ++it1)
        {
            res += *it1;
        }

        // emit the result
        // std::cout << "hi ";
        // std::cout << (int)key << " " << res << "\n";
        runtime.emit(key, res);
    }
};

typedef
mapreduce::job<hyperlink::map_task,
               hyperlink::reduce_task,
               mapreduce::null_combiner,
               hyperlink::datasource<hyperlink::map_task>
> job;

} // namespace hyperlink

int main(int argc, char *argv[])
{
    // looking into input file to get size of required pagerank matrix.
    std::ifstream fin;
    fin.open("../pagerank-master/test/barabasi-20000.txt");
    bool debug = 0;
    int fr,to,max=0;
    std::string s;
    while(fin)
    {
        fin >> fr >> to;
        max = std::max(max,to);
        max = std::max(max,fr);
    }
    fin.close();
    max++;
    // found and stored in max # pages on network.
    hyperlink::n = max;
    // initializing transpose of graph matrix for only non-zero entry.
    for(int i=0;i<max;i++)
    {
        std::vector<int> temp;
        hyperlink::links.push_back(temp);
    }
    fin.open("../pagerank-master/test/barabasi-20000.txt");
    while(fin)
    {
        fin >> fr >> to;
        hyperlink::links[fr].push_back(to);
    }
    fin.close();
    double alpha = 0.85;
    // initializing A matric corresponds to dangling nodes (only stored single value per column of H matrix)
    std::vector<double> A;
    for(int i=0;i<hyperlink::links.size();i++)
    { 
        if(hyperlink::links[i].size()==0)
            A.push_back(1.0/max);
        else
            A.push_back(0);
    }
    std::cout << "\n";
    // initializing I0 (eign vector)
    for(int i=0;i<max;i++)
        hyperlink::eign.push_back(1.0/max);
    mapreduce::specification spec;
    if (argc > 1)
        spec.map_tasks = std::max(1, atoi(argv[1]));
    if (argc > 2)
        spec.reduce_tasks = atoi(argv[2]);
    else
        spec.reduce_tasks = std::max(1U, std::thread::hardware_concurrency());
    std::cout << "preprocessing done!\n";
    int count =0;
    clock_t time = clock();
    while(true)
    {
        // std::cout<<"hi\n";
        std::cout << count << " " << hyperlink::eign[0] << "\n";
        count++;
        hyperlink::job::datasource_type datasource;
        hyperlink::job job(datasource, spec);
        mapreduce::results result;
        bool b=1;
        if(debug)
            job.run<mapreduce::schedule_policy::sequential<hyperlink::job> >(result);
        else
            // parallel call to map reduce function in hyperlink which is define above.
            job.run<mapreduce::schedule_policy::cpu_parallel<hyperlink::job> >(result);

                
        std::vector<double> temp(hyperlink::eign.begin(),hyperlink::eign.end());
        for(int i=0;i<max;i++)
            hyperlink::eign[i] = 0;
        //alpha*H*I
        for (auto it=job.begin_results(); it!=job.end_results(); ++it)
        {
            hyperlink::eign[it->first]=alpha*(it->second);
        }
        //(alpha*A+(1-alpha)/n*1)*I
        double add=0.0;
        for(int i=0;i<max;i++)
        {
            add += alpha*A[i]*temp[i] + (1-alpha)*temp[i]/max; 
        }
        // std::cout << add <<"\n";
        for(int i=0;i<max;i++)
            hyperlink::eign[i] += add;
        double err = 0.0;
        for(int i=0;i<max;i++)
        {
            double t = std::abs(hyperlink::eign[i]-temp[i]);
            err += t;
            if(t>0.00001)
                b=0;
        }
        if(err<0.000001)
            break;
    }
    // std::cout << 2.0/max <<"\n";
    std::cout << "\n" << count << " iterations in " <<(double)(clock()-time)/CLOCKS_PER_SEC <<" seconds\n";

    fin.open("../pagerank-master/test/barabasi-20000-pr-j.txt");
    std::string s1,s2;
    double f,fmax=0,tot=0.0;
    for(int i=0;i<max;i++)
    {
        fin >> s1 >> s2 >> f;
        double temp = std::abs(hyperlink::eign[i]-f);
        tot += temp;
        fmax = std::max(fmax,temp);
        // std::cout << temp <<" ";
    }
    std::cout<<"max error = "<<fmax<<"\nsum error = "<< tot << "\n";
    fin.close();
    return 0;
}

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

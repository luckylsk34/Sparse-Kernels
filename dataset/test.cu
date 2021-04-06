//this file is for testing all the cuda funcion
// one whlch contains main()
#include<stdio.h>
#include"properties.h"

int main(){
    printf("hello world\n");
    //calling a function  form matrix.cu  for testing
    // test(45);
    int *t;
    
    size_t size = 32*32*sizeof(int);

	t = (int*)malloc(size);

    //matrix(t,(uint)32,(uint)32,(uint)32,(uint)32);
    for(int i=0;i<32;i++){
        for(int j=0;j<32;j++){
            //t[i*32+j] = (rand()%255)*(rand()%2);
            t[i*32+j] = 1;
        }
        //printf("\n");
    }


    for(int i=0;i<32;i++){
        for(int j=0;j<32;j++){
            printf("%d ",t[i*32+j]);
        }
        printf("\n");
    }

    printf("density = %lf\n",density(t,32,32));
    printf("sparisity = %lf\n",sparisity(t,32,32));
    printf("issparse = %d\n",issparse(t,32,32));
    printf("mean = %lf\n",mean(t,32,32));
    
    double* res;
    res = (double*)malloc(32*sizeof(double));
    row_mean(t,32,32,res);
    printf("row mean\n");
    for(int i=0;i<32;i++){
        printf("%lf ",res[i]);
    }
    printf("\n");

    double* res1;
    res1 = (double*)malloc(3*sizeof(double));
    col_mean(t,3,3,res);
    printf("col mean\n");
    for(int i=0;i<3;i++){
        printf("%lf ",res1[i]);
    }
    printf("\n");
}
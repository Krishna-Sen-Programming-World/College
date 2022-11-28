#include<stdio.h>
int main(){
    int arr[1000];
    printf("ENTER THE SIZE OF THE ARRAY ");
    int n;
    scanf("%d",&n);
    printf("\nENTER THE ELEMENTS ");
    for(int i=0;i<n;i++)
    scanf("%d",&arr[i]);
    for(int i=0;i<n-1;i++)
    for(int j=0;j<n-1-i;j++)
    if(arr[j]>arr[j+1]){
        arr[j]=arr[j]^arr[j+1];
        arr[j+1]=arr[j]^arr[j+1];
        arr[j]=arr[j]^arr[j+1];
    }
    printf("\nTHE ELEMENTS AFTER ASCENDING ORDER IS ");
    for(int i=0;i<n;i++)
    printf("%d ",arr[i]);
}

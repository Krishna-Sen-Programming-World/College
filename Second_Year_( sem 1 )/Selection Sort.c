#include<stdio.h>
int main(){
    int arr[1000],n;
    printf("ENTER THE SIZE OF THE ARRAY ");
    scanf("%d",&n);
    printf("ENTER THE ELEMENTS ");
    for(int i=0;i<n;i++)
    scanf("%d",&arr[i]);
    for(int i=0;i<n-1;i++){
        int min=i;
        for(int j=i+1;j<n;j++)
        if(arr[min]>arr[j]){
            min=j;
        }
        int temp=arr[min];
        arr[min]=arr[i];
        arr[i]=temp;
    }
    printf("\nTHE ARRAY AFTER SORTING IS ");
    for(int i=0;i<n;i++)
    printf("%d ",arr[i]);
    return 0;
}

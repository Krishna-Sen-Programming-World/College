#include <stdio.h>
int main()
{
    int s;
    printf("Enter the size of the matrix : ");
    scanf("%d",&s);
    int arr[s][s];
    
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++){
            if(i==j)
            arr[i][j]=0;
            else if(i<j)
            arr[i][j]=1;
            else
            arr[i][j]=-1;
        }
    }
    printf("\nThe elements of the matrix are\n");
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++)
        printf("\t%d",arr[i][j]);
        printf("\n");
    }
    return 0;
}

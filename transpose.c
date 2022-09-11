#include <stdio.h>
int main()
{
    int s;
    printf("Enter the size of the matrix : ");
    scanf("%d",&s);
    int arr[s][s];
    
    printf("\nEnter the elements of the matrix");
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++)
        scanf("%d",&arr[i][j]);
    }
    
    printf("\nThe elements of the matrix before transpose is \n");
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++)
        printf("\t%d",arr[i][j]);
        printf("\n");
    }
    printf("\nThe elements of the matrix after transpose is \n");
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++)
        printf("\t%d",arr[j][i]);
        printf("\n");
    }

    return 0;
}

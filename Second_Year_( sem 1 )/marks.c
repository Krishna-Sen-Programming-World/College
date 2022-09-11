#include <stdio.h>
int main()
{
    int arr[5][3];
    printf("Enter the student the marks:\n");
    for(int i=0;i<5;i++){
        for(int j=0;j<3;j++)
        scanf("%d",&arr[i][j]);
    }
    printf("\nThe student the marks are\n");
    for(int i=0;i<5;i++){
        for(int j=0;j<3;j++)
        printf("\t%d",arr[i][j]);
        printf("\n");
    }
    
    for(int i=0;i<3;i++){
        int max=arr[0][i];
        int id=0;
        for(int j=1;j<5;j++){
            if(max<arr[j][i]){
                max=arr[j][i];
                id=j;
            }
        }
        printf("\nindex[%d] student got max marks in subject no:%d which is %d",id,i,max);
    }
}

#include <stdio.h>

int size,top=-1,bottom=0,count=-1;
int arr[1000];

void push(){
    if(count==size-1)
    printf("Overflow\n");
    else{
        int a;
        printf("Enter the value ");
        scanf("%d",&a);
        top=(top+1)%size;
        arr[top]=a;
        count++;
    }
}

void pop(){
    if(count==-1)
    printf("Underflow\n");
    else
    {
        printf("The popped value is %d\n",arr[bottom]);
        bottom=(bottom+1)%size;
        count--;
    }
}

void display(){
    printf("The array is ");
    for(int i=0,j=bottom;i<=count;i++,j=((j+1)%size))
    printf(" %d",arr[j]);
    printf("\n");
}


int main()
{
    printf("Enter the size of array ");
    scanf("%d",&size);
    for(int f=0;f!=1;){
        printf("1 for push 2 for pop 3 for display ");
        int ch;
        scanf("%d",&ch);
        switch(ch){
            case 1:
            push();
            break;
            case 2:
            pop();
            break;
            case 3:
            display();
            break;
        }
    }
    return 0;
}

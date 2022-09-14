#include <stdio.h>
int top=-1,size;
void push(int arr[]){
    if(top==size-1)
    printf("Overflow\n");
    else{
        int value;
        printf("Enter the value\n");
        scanf("%d",&value);
        arr[++top]=value;
    }
}

void pop(int arr[]){
    if(top==-1)
    printf("Stack is empty\n");
    else
    printf("%d\n",arr[top--]);
}

void display(int arr[]){
    for(int i=0;i<=top;i++)
    printf("%d ",arr[i]);
    printf("\n");
}
int main()
{
    int ch, sizes;
    printf("Enter the size of the array\n");
    scanf("%d",&sizes);
    size=sizes;
    int arr[size];
    printf("Enter the choice:\n1 for insertion\n2 for deletion\n3 for display\n");
    scanf("%d",&ch);
    while(ch<=3)
    {
        switch(ch){
            case 1:
            push(arr);
            printf("Enter the choice:\n1 for insertion\n2 for deletion\n3 for display\n");
            scanf("%d",&ch);
            break;
            case 2:
            pop(arr);
            printf("Enter the choice:\n1 for insertion\n2 for deletion\n3 for display\n");
            scanf("%d",&ch);
            break;
            case 3:
            display(arr);
            printf("Enter the choice:\n1 for insertion\n2 for deletion\n3 for display\n");
            scanf("%d",&ch);
            break;
            default:
            printf("Enter the right choice\n");
        }
    }
    return 0;
}

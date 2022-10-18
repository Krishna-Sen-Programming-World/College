#include<stdio.h>
int stack[100];
int queue[100];
int top=-1,size;
void push_s()
{
    if(top==size)
    printf("Overflow");
    else{
        int x;
        printf("\nEnter the value ");
        scanf("%d",&x);
        stack[++top]=x;
    }
}


void push_q(){
    for(int i=0,j=top;j>=0;j--,i++)
    queue[i]=stack[j];
}


void push_qstack(){
    for(int i=0;i<top;i++)
    stack[i]=queue[i+1];
    top--;
    printf("\nThe stack now is : ");
    for(int i=0;i<=top;i++)
    printf("%d ",stack[i]);

}


void pop_queue(){
    if(top==-1)
    printf("underflow");
    else{
        push_q();
        printf("Value popped is %d",queue[0]);
        push_qstack();
    }
}

int main()
{
    printf("Enter the size ");
    scanf("%d",&size);
    // size=x;
    int ch;
    for(int i=0;i!=1;){
        printf("\n1 for push 2 for pop 3 for exit ");
        scanf("%d",&ch);
        switch(ch){
            case 1:
            push_s();
            break;
            case 2:
            pop_queue();
            break;
            case 3:
            i=1;
            break;
            default:
            printf("Enter the right choice");
            
        }
    }
}

#include <stdio.h>

#include "DataStructures/c_lib/clist.h"

int main(void)
{
    List_int* l = create_with_same_values_List_int(10, 12);
    printf("%d\n", get_List_int(l, 9));
    destroy_List_int(l);
    return 0;
}
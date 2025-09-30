#include <stdio.h>
#include <stdlib.h>
#include "error_ffi.h"
#include "object_store_ffi.h"
#include "object_store_writer_ffi.h"
#include "test_ffi.h"

int main() {
    ObjectStoreWrapper* store = NULL;
    ObjectStoreWriterWrapper* oswriter = NULL;
    C_BRIDGE_STATUS result = C_SUCCESS;
    
    // let endpoint = "http://localhost:9000";
    // let access_key_id = "minioadmin";
    // let secret_access_key = "minioadmin";
    // let region = "";
    // let bucket_name = "rust-bucket";

    result = create_object_store(
        "s3",
        "http://localhost:9000",
        "minioadmin",
        "minioadmin",
        "",
        "rust-bucket",
        &store
    );
    
    if (result != 0) {
        printf("Failed to create S3 object store: %d\n", result);
        return -1;
    }

    result = create_object_store_writer(
        store,
        "test_file.vx",
        &oswriter);

    if (result != 0) {
        printf("Failed to create S3 object store writer: %d\n", result);
        return -1;
    }
    free_object_store_writer(oswriter);
    
    result = test_bridge_object_store_writer(store, "test_file.vx");
    if (result != 0) {
        printf("Failed to write S3 object: %d\n", result);
        return -1;
    }

    free_object_store_wrapper(store);
    printf("PASS\n");

    return 0;
}

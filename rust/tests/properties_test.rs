use milvus_storage_datafusion::PropertiesBuilder;

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_empty() {
    let builder = PropertiesBuilder::new();
    let properties = builder.build().expect("Should build empty properties successfully");
    
    // Empty properties should have null pointer and zero count
    assert!(properties.properties.is_null());
    assert_eq!(properties.count, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_single_property() {
    let builder = PropertiesBuilder::new();
    let properties = builder
        .add_property("test_key", "test_value")
        .expect("Should add property successfully")
        .build()
        .expect("Should build properties successfully");
    
    // Should have non-null pointer and count of 1
    assert!(!properties.properties.is_null());
    assert_eq!(properties.count, 1);
    
    // Test getting the property value
    let value = properties.get("test_key");
    assert_eq!(value, Some("test_value".to_string()));
    
    // Test getting non-existent property
    let missing = properties.get("missing_key");
    assert_eq!(missing, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_multiple_properties() {
    let builder = PropertiesBuilder::new();
    let properties = builder
        .add_property("fs.storage_type", "local")
        .expect("Should add storage_type property")
        .add_property("fs.root_path", "/tmp/test")
        .expect("Should add root_path property")
        .add_property("fs.use_ssl", "false")
        .expect("Should add use_ssl property")
        .build()
        .expect("Should build properties successfully");
    
    // Should have non-null pointer and count of 3
    assert!(!properties.properties.is_null());
    assert_eq!(properties.count, 3);
    
    // Test all property values
    assert_eq!(properties.get("fs.storage_type"), Some("local".to_string()));
    assert_eq!(properties.get("fs.root_path"), Some("/tmp/test".to_string()));
    assert_eq!(properties.get("fs.use_ssl"), Some("false".to_string()));
    
    // Test non-existent property
    assert_eq!(properties.get("non_existent"), None);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_special_characters() {
    let builder = PropertiesBuilder::new();
    let properties = builder
        .add_property("key_with_spaces", "value with spaces")
        .expect("Should handle spaces")
        .add_property("key.with.dots", "value.with.dots")
        .expect("Should handle dots")
        .add_property("key-with-dashes", "value-with-dashes")
        .expect("Should handle dashes")
        .add_property("key_with_numbers123", "value_with_numbers456")
        .expect("Should handle numbers")
        .build()
        .expect("Should build properties successfully");
    
    assert_eq!(properties.count, 4);
    assert_eq!(properties.get("key_with_spaces"), Some("value with spaces".to_string()));
    assert_eq!(properties.get("key.with.dots"), Some("value.with.dots".to_string()));
    assert_eq!(properties.get("key-with-dashes"), Some("value-with-dashes".to_string()));
    assert_eq!(properties.get("key_with_numbers123"), Some("value_with_numbers456".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_empty_values() {
    let builder = PropertiesBuilder::new();
    let properties = builder
        .add_property("empty_key", "")
        .expect("Should handle empty value")
        .add_property("normal_key", "normal_value")
        .expect("Should handle normal value")
        .build()
        .expect("Should build properties successfully");
    
    assert_eq!(properties.count, 2);
    assert_eq!(properties.get("empty_key"), Some("".to_string()));
    assert_eq!(properties.get("normal_key"), Some("normal_value".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_invalid_strings() {
    let builder = PropertiesBuilder::new();
    
    // Test key with null byte (should fail)
    let result = builder.add_property("key\0with_null", "value");
    assert!(result.is_err());
    
    let builder = PropertiesBuilder::new();
    // Test value with null byte (should fail)
    let result = builder.add_property("key", "value\0with_null");
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_builder_chaining() {
    // Test that the builder pattern works correctly with method chaining
    let properties = PropertiesBuilder::new()
        .add_property("key1", "value1")
        .expect("Should add key1")
        .add_property("key2", "value2")
        .expect("Should add key2")
        .add_property("key3", "value3")
        .expect("Should add key3")
        .build()
        .expect("Should build properties");
    
    assert_eq!(properties.count, 3);
    assert_eq!(properties.get("key1"), Some("value1".to_string()));
    assert_eq!(properties.get("key2"), Some("value2".to_string()));
    assert_eq!(properties.get("key3"), Some("value3".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_drop_cleanup() {
    for i in 0..10 {
        let properties = PropertiesBuilder::new()
            .add_property(&format!("key_{}", i), &format!("value_{}", i))
            .expect("Should add property")
            .build()
            .expect("Should build properties");
        
        assert_eq!(properties.count, 1);
        assert_eq!(properties.get(&format!("key_{}", i)), Some(format!("value_{}", i)));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_filesystem_config() {
    // Test the specific properties that were causing crashes
    let properties = PropertiesBuilder::new()
        .add_property("fs.storage_type", "local")
        .expect("Should add storage_type")
        .add_property("fs.root_path", "/tmp/")
        .expect("Should add root_path")
        .build()
        .expect("Should build filesystem properties");
    
    assert_eq!(properties.count, 2);
    assert_eq!(properties.get("fs.storage_type"), Some("local".to_string()));
    assert_eq!(properties.get("fs.root_path"), Some("/tmp/".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_properties_concurrent_access() {
    // Test that properties can be safely accessed from multiple threads
    use std::sync::Arc;
    use tokio::task;
    
    let properties = Arc::new(
        PropertiesBuilder::new()
            .add_property("shared_key", "shared_value")
            .expect("Should add property")
            .build()
            .expect("Should build properties")
    );
    
    let mut handles = vec![];
    
    // Spawn multiple tasks that access the properties concurrently
    for i in 0..5 {
        let props = properties.clone();
        let handle = task::spawn(async move {
            let value = props.get("shared_key");
            assert_eq!(value, Some("shared_value".to_string()));
            i // Return task id for verification
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.expect("Task should complete successfully");
        assert_eq!(result, i);
    }
}

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.31.0
// 	protoc        v3.21.9
// source: manifest.proto

package manifest_proto

import (
	schema_proto "github.com/milvus-io/milvus-storage/go/proto/schema_proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type Options struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Uri string `protobuf:"bytes,1,opt,name=uri,proto3" json:"uri,omitempty"`
}

func (x *Options) Reset() {
	*x = Options{}
	if protoimpl.UnsafeEnabled {
		mi := &file_manifest_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Options) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Options) ProtoMessage() {}

func (x *Options) ProtoReflect() protoreflect.Message {
	mi := &file_manifest_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Options.ProtoReflect.Descriptor instead.
func (*Options) Descriptor() ([]byte, []int) {
	return file_manifest_proto_rawDescGZIP(), []int{0}
}

func (x *Options) GetUri() string {
	if x != nil {
		return x.Uri
	}
	return ""
}

type Manifest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Version         int64                `protobuf:"varint,1,opt,name=version,proto3" json:"version,omitempty"`
	Options         *Options             `protobuf:"bytes,2,opt,name=options,proto3" json:"options,omitempty"`
	Schema          *schema_proto.Schema `protobuf:"bytes,3,opt,name=schema,proto3" json:"schema,omitempty"`
	ScalarFragments []*Fragment          `protobuf:"bytes,4,rep,name=scalar_fragments,json=scalarFragments,proto3" json:"scalar_fragments,omitempty"`
	VectorFragments []*Fragment          `protobuf:"bytes,5,rep,name=vector_fragments,json=vectorFragments,proto3" json:"vector_fragments,omitempty"`
	DeleteFragments []*Fragment          `protobuf:"bytes,6,rep,name=delete_fragments,json=deleteFragments,proto3" json:"delete_fragments,omitempty"`
	Blobs           []*Blob              `protobuf:"bytes,7,rep,name=blobs,proto3" json:"blobs,omitempty"`
}

func (x *Manifest) Reset() {
	*x = Manifest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_manifest_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Manifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Manifest) ProtoMessage() {}

func (x *Manifest) ProtoReflect() protoreflect.Message {
	mi := &file_manifest_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Manifest.ProtoReflect.Descriptor instead.
func (*Manifest) Descriptor() ([]byte, []int) {
	return file_manifest_proto_rawDescGZIP(), []int{1}
}

func (x *Manifest) GetVersion() int64 {
	if x != nil {
		return x.Version
	}
	return 0
}

func (x *Manifest) GetOptions() *Options {
	if x != nil {
		return x.Options
	}
	return nil
}

func (x *Manifest) GetSchema() *schema_proto.Schema {
	if x != nil {
		return x.Schema
	}
	return nil
}

func (x *Manifest) GetScalarFragments() []*Fragment {
	if x != nil {
		return x.ScalarFragments
	}
	return nil
}

func (x *Manifest) GetVectorFragments() []*Fragment {
	if x != nil {
		return x.VectorFragments
	}
	return nil
}

func (x *Manifest) GetDeleteFragments() []*Fragment {
	if x != nil {
		return x.DeleteFragments
	}
	return nil
}

func (x *Manifest) GetBlobs() []*Blob {
	if x != nil {
		return x.Blobs
	}
	return nil
}

type Fragment struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id    int64    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	Files []string `protobuf:"bytes,2,rep,name=files,proto3" json:"files,omitempty"`
}

func (x *Fragment) Reset() {
	*x = Fragment{}
	if protoimpl.UnsafeEnabled {
		mi := &file_manifest_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Fragment) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fragment) ProtoMessage() {}

func (x *Fragment) ProtoReflect() protoreflect.Message {
	mi := &file_manifest_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Fragment.ProtoReflect.Descriptor instead.
func (*Fragment) Descriptor() ([]byte, []int) {
	return file_manifest_proto_rawDescGZIP(), []int{2}
}

func (x *Fragment) GetId() int64 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *Fragment) GetFiles() []string {
	if x != nil {
		return x.Files
	}
	return nil
}

type Blob struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Size int64  `protobuf:"varint,2,opt,name=size,proto3" json:"size,omitempty"`
	File string `protobuf:"bytes,3,opt,name=file,proto3" json:"file,omitempty"`
}

func (x *Blob) Reset() {
	*x = Blob{}
	if protoimpl.UnsafeEnabled {
		mi := &file_manifest_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Blob) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Blob) ProtoMessage() {}

func (x *Blob) ProtoReflect() protoreflect.Message {
	mi := &file_manifest_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Blob.ProtoReflect.Descriptor instead.
func (*Blob) Descriptor() ([]byte, []int) {
	return file_manifest_proto_rawDescGZIP(), []int{3}
}

func (x *Blob) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Blob) GetSize() int64 {
	if x != nil {
		return x.Size
	}
	return 0
}

func (x *Blob) GetFile() string {
	if x != nil {
		return x.File
	}
	return ""
}

var File_manifest_proto protoreflect.FileDescriptor

var file_manifest_proto_rawDesc = []byte{
	0x0a, 0x0e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x12, 0x0e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x1a, 0x0c, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x1b,
	0x0a, 0x07, 0x4f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x12, 0x10, 0x0a, 0x03, 0x75, 0x72, 0x69,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x75, 0x72, 0x69, 0x22, 0x80, 0x03, 0x0a, 0x08,
	0x4d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x12, 0x18, 0x0a, 0x07, 0x76, 0x65, 0x72, 0x73,
	0x69, 0x6f, 0x6e, 0x18, 0x01, 0x20, 0x01, 0x28, 0x03, 0x52, 0x07, 0x76, 0x65, 0x72, 0x73, 0x69,
	0x6f, 0x6e, 0x12, 0x31, 0x0a, 0x07, 0x6f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x5f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x4f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x52, 0x07, 0x6f, 0x70,
	0x74, 0x69, 0x6f, 0x6e, 0x73, 0x12, 0x2c, 0x0a, 0x06, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x18,
	0x03, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x5f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x53, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x52, 0x06, 0x73, 0x63, 0x68,
	0x65, 0x6d, 0x61, 0x12, 0x43, 0x0a, 0x10, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x5f, 0x66, 0x72,
	0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x18, 0x04, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x18, 0x2e,
	0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x46,
	0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x52, 0x0f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x46,
	0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x12, 0x43, 0x0a, 0x10, 0x76, 0x65, 0x63, 0x74,
	0x6f, 0x72, 0x5f, 0x66, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x18, 0x05, 0x20, 0x03,
	0x28, 0x0b, 0x32, 0x18, 0x2e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x5f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x2e, 0x46, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x52, 0x0f, 0x76, 0x65,
	0x63, 0x74, 0x6f, 0x72, 0x46, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x12, 0x43, 0x0a,
	0x10, 0x64, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x5f, 0x66, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74,
	0x73, 0x18, 0x06, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x18, 0x2e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65,
	0x73, 0x74, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x46, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e,
	0x74, 0x52, 0x0f, 0x64, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x46, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e,
	0x74, 0x73, 0x12, 0x2a, 0x0a, 0x05, 0x62, 0x6c, 0x6f, 0x62, 0x73, 0x18, 0x07, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x5f, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x2e, 0x42, 0x6c, 0x6f, 0x62, 0x52, 0x05, 0x62, 0x6c, 0x6f, 0x62, 0x73, 0x22, 0x30,
	0x0a, 0x08, 0x46, 0x72, 0x61, 0x67, 0x6d, 0x65, 0x6e, 0x74, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x03, 0x52, 0x02, 0x69, 0x64, 0x12, 0x14, 0x0a, 0x05, 0x66, 0x69,
	0x6c, 0x65, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x09, 0x52, 0x05, 0x66, 0x69, 0x6c, 0x65, 0x73,
	0x22, 0x42, 0x0a, 0x04, 0x42, 0x6c, 0x6f, 0x62, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x12, 0x0a, 0x04,
	0x73, 0x69, 0x7a, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x03, 0x52, 0x04, 0x73, 0x69, 0x7a, 0x65,
	0x12, 0x12, 0x0a, 0x04, 0x66, 0x69, 0x6c, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04,
	0x66, 0x69, 0x6c, 0x65, 0x42, 0x41, 0x5a, 0x3f, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63,
	0x6f, 0x6d, 0x2f, 0x6d, 0x69, 0x6c, 0x76, 0x75, 0x73, 0x2d, 0x69, 0x6f, 0x2f, 0x6d, 0x69, 0x6c,
	0x76, 0x75, 0x73, 0x2d, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x2d, 0x66, 0x6f, 0x72, 0x6d,
	0x61, 0x74, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73,
	0x74, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_manifest_proto_rawDescOnce sync.Once
	file_manifest_proto_rawDescData = file_manifest_proto_rawDesc
)

func file_manifest_proto_rawDescGZIP() []byte {
	file_manifest_proto_rawDescOnce.Do(func() {
		file_manifest_proto_rawDescData = protoimpl.X.CompressGZIP(file_manifest_proto_rawDescData)
	})
	return file_manifest_proto_rawDescData
}

var file_manifest_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_manifest_proto_goTypes = []interface{}{
	(*Options)(nil),             // 0: manifest_proto.Options
	(*Manifest)(nil),            // 1: manifest_proto.Manifest
	(*Fragment)(nil),            // 2: manifest_proto.Fragment
	(*Blob)(nil),                // 3: manifest_proto.Blob
	(*schema_proto.Schema)(nil), // 4: schema_proto.Schema
}
var file_manifest_proto_depIdxs = []int32{
	0, // 0: manifest_proto.Manifest.options:type_name -> manifest_proto.Options
	4, // 1: manifest_proto.Manifest.schema:type_name -> schema_proto.Schema
	2, // 2: manifest_proto.Manifest.scalar_fragments:type_name -> manifest_proto.Fragment
	2, // 3: manifest_proto.Manifest.vector_fragments:type_name -> manifest_proto.Fragment
	2, // 4: manifest_proto.Manifest.delete_fragments:type_name -> manifest_proto.Fragment
	3, // 5: manifest_proto.Manifest.blobs:type_name -> manifest_proto.Blob
	6, // [6:6] is the sub-list for method output_type
	6, // [6:6] is the sub-list for method input_type
	6, // [6:6] is the sub-list for extension type_name
	6, // [6:6] is the sub-list for extension extendee
	0, // [0:6] is the sub-list for field type_name
}

func init() { file_manifest_proto_init() }
func file_manifest_proto_init() {
	if File_manifest_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_manifest_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Options); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_manifest_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Manifest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_manifest_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Fragment); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_manifest_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Blob); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_manifest_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_manifest_proto_goTypes,
		DependencyIndexes: file_manifest_proto_depIdxs,
		MessageInfos:      file_manifest_proto_msgTypes,
	}.Build()
	File_manifest_proto = out.File
	file_manifest_proto_rawDesc = nil
	file_manifest_proto_goTypes = nil
	file_manifest_proto_depIdxs = nil
}

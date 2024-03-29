// Code generated by protoc-gen-go. DO NOT EDIT.
// source: manifest.proto

package manifest_proto

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	schema_proto "github.com/milvus-io/milvus-storage/go/proto/schema_proto"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type Options struct {
	Uri                  string   `protobuf:"bytes,1,opt,name=uri,proto3" json:"uri,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Options) Reset()         { *m = Options{} }
func (m *Options) String() string { return proto.CompactTextString(m) }
func (*Options) ProtoMessage()    {}
func (*Options) Descriptor() ([]byte, []int) {
	return fileDescriptor_0bb23f43f7afb4c1, []int{0}
}

func (m *Options) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Options.Unmarshal(m, b)
}
func (m *Options) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Options.Marshal(b, m, deterministic)
}
func (m *Options) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Options.Merge(m, src)
}
func (m *Options) XXX_Size() int {
	return xxx_messageInfo_Options.Size(m)
}
func (m *Options) XXX_DiscardUnknown() {
	xxx_messageInfo_Options.DiscardUnknown(m)
}

var xxx_messageInfo_Options proto.InternalMessageInfo

func (m *Options) GetUri() string {
	if m != nil {
		return m.Uri
	}
	return ""
}

type Manifest struct {
	Version              int64                `protobuf:"varint,1,opt,name=version,proto3" json:"version,omitempty"`
	Options              *Options             `protobuf:"bytes,2,opt,name=options,proto3" json:"options,omitempty"`
	Schema               *schema_proto.Schema `protobuf:"bytes,3,opt,name=schema,proto3" json:"schema,omitempty"`
	ScalarFragments      []*Fragment          `protobuf:"bytes,4,rep,name=scalar_fragments,json=scalarFragments,proto3" json:"scalar_fragments,omitempty"`
	VectorFragments      []*Fragment          `protobuf:"bytes,5,rep,name=vector_fragments,json=vectorFragments,proto3" json:"vector_fragments,omitempty"`
	DeleteFragments      []*Fragment          `protobuf:"bytes,6,rep,name=delete_fragments,json=deleteFragments,proto3" json:"delete_fragments,omitempty"`
	Blobs                []*Blob              `protobuf:"bytes,7,rep,name=blobs,proto3" json:"blobs,omitempty"`
	XXX_NoUnkeyedLiteral struct{}             `json:"-"`
	XXX_unrecognized     []byte               `json:"-"`
	XXX_sizecache        int32                `json:"-"`
}

func (m *Manifest) Reset()         { *m = Manifest{} }
func (m *Manifest) String() string { return proto.CompactTextString(m) }
func (*Manifest) ProtoMessage()    {}
func (*Manifest) Descriptor() ([]byte, []int) {
	return fileDescriptor_0bb23f43f7afb4c1, []int{1}
}

func (m *Manifest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Manifest.Unmarshal(m, b)
}
func (m *Manifest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Manifest.Marshal(b, m, deterministic)
}
func (m *Manifest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Manifest.Merge(m, src)
}
func (m *Manifest) XXX_Size() int {
	return xxx_messageInfo_Manifest.Size(m)
}
func (m *Manifest) XXX_DiscardUnknown() {
	xxx_messageInfo_Manifest.DiscardUnknown(m)
}

var xxx_messageInfo_Manifest proto.InternalMessageInfo

func (m *Manifest) GetVersion() int64 {
	if m != nil {
		return m.Version
	}
	return 0
}

func (m *Manifest) GetOptions() *Options {
	if m != nil {
		return m.Options
	}
	return nil
}

func (m *Manifest) GetSchema() *schema_proto.Schema {
	if m != nil {
		return m.Schema
	}
	return nil
}

func (m *Manifest) GetScalarFragments() []*Fragment {
	if m != nil {
		return m.ScalarFragments
	}
	return nil
}

func (m *Manifest) GetVectorFragments() []*Fragment {
	if m != nil {
		return m.VectorFragments
	}
	return nil
}

func (m *Manifest) GetDeleteFragments() []*Fragment {
	if m != nil {
		return m.DeleteFragments
	}
	return nil
}

func (m *Manifest) GetBlobs() []*Blob {
	if m != nil {
		return m.Blobs
	}
	return nil
}

type Fragment struct {
	Id                   int64    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	Files                []string `protobuf:"bytes,2,rep,name=files,proto3" json:"files,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Fragment) Reset()         { *m = Fragment{} }
func (m *Fragment) String() string { return proto.CompactTextString(m) }
func (*Fragment) ProtoMessage()    {}
func (*Fragment) Descriptor() ([]byte, []int) {
	return fileDescriptor_0bb23f43f7afb4c1, []int{2}
}

func (m *Fragment) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Fragment.Unmarshal(m, b)
}
func (m *Fragment) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Fragment.Marshal(b, m, deterministic)
}
func (m *Fragment) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Fragment.Merge(m, src)
}
func (m *Fragment) XXX_Size() int {
	return xxx_messageInfo_Fragment.Size(m)
}
func (m *Fragment) XXX_DiscardUnknown() {
	xxx_messageInfo_Fragment.DiscardUnknown(m)
}

var xxx_messageInfo_Fragment proto.InternalMessageInfo

func (m *Fragment) GetId() int64 {
	if m != nil {
		return m.Id
	}
	return 0
}

func (m *Fragment) GetFiles() []string {
	if m != nil {
		return m.Files
	}
	return nil
}

type Blob struct {
	Name                 string   `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Size                 int64    `protobuf:"varint,2,opt,name=size,proto3" json:"size,omitempty"`
	File                 string   `protobuf:"bytes,3,opt,name=file,proto3" json:"file,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Blob) Reset()         { *m = Blob{} }
func (m *Blob) String() string { return proto.CompactTextString(m) }
func (*Blob) ProtoMessage()    {}
func (*Blob) Descriptor() ([]byte, []int) {
	return fileDescriptor_0bb23f43f7afb4c1, []int{3}
}

func (m *Blob) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Blob.Unmarshal(m, b)
}
func (m *Blob) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Blob.Marshal(b, m, deterministic)
}
func (m *Blob) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Blob.Merge(m, src)
}
func (m *Blob) XXX_Size() int {
	return xxx_messageInfo_Blob.Size(m)
}
func (m *Blob) XXX_DiscardUnknown() {
	xxx_messageInfo_Blob.DiscardUnknown(m)
}

var xxx_messageInfo_Blob proto.InternalMessageInfo

func (m *Blob) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *Blob) GetSize() int64 {
	if m != nil {
		return m.Size
	}
	return 0
}

func (m *Blob) GetFile() string {
	if m != nil {
		return m.File
	}
	return ""
}

func init() {
	proto.RegisterType((*Options)(nil), "manifest_proto.Options")
	proto.RegisterType((*Manifest)(nil), "manifest_proto.Manifest")
	proto.RegisterType((*Fragment)(nil), "manifest_proto.Fragment")
	proto.RegisterType((*Blob)(nil), "manifest_proto.Blob")
}

func init() { proto.RegisterFile("manifest.proto", fileDescriptor_0bb23f43f7afb4c1) }

var fileDescriptor_0bb23f43f7afb4c1 = []byte{
	// 343 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x91, 0xcf, 0x4f, 0xc2, 0x30,
	0x1c, 0xc5, 0x03, 0x05, 0x06, 0x5f, 0x0c, 0x92, 0x86, 0xc4, 0x46, 0x2f, 0x64, 0xa7, 0xc5, 0xe8,
	0xa6, 0x78, 0x34, 0x5e, 0x30, 0xf1, 0x66, 0x4c, 0xea, 0xcd, 0x0b, 0xe9, 0x46, 0x19, 0x4d, 0xb6,
	0x95, 0xac, 0x85, 0x83, 0x27, 0xff, 0x74, 0xd3, 0x1f, 0x53, 0xc6, 0x85, 0x13, 0xef, 0x95, 0xf7,
	0x3e, 0xdf, 0xae, 0x5f, 0x98, 0x94, 0xac, 0x12, 0x1b, 0xae, 0x74, 0xbc, 0xab, 0xa5, 0x96, 0xf8,
	0xcf, 0xaf, 0xac, 0xbf, 0xbe, 0x50, 0xd9, 0x96, 0x97, 0xcc, 0xfd, 0x1b, 0xde, 0x40, 0xf0, 0xb1,
	0xd3, 0x42, 0x56, 0x0a, 0x4f, 0x01, 0xed, 0x6b, 0x41, 0x3a, 0xf3, 0x4e, 0x34, 0xa2, 0x46, 0x86,
	0x3f, 0x08, 0x86, 0xef, 0xbe, 0x8d, 0x09, 0x04, 0x07, 0x5e, 0x2b, 0x21, 0x2b, 0x1b, 0x41, 0xb4,
	0xb1, 0xf8, 0x11, 0x02, 0xe9, 0x18, 0xa4, 0x3b, 0xef, 0x44, 0xe3, 0xc5, 0x55, 0xdc, 0x9e, 0x19,
	0xfb, 0x11, 0xb4, 0xc9, 0xe1, 0x3b, 0x18, 0xb8, 0x6b, 0x10, 0x64, 0x1b, 0xb3, 0xd8, 0x59, 0x9f,
	0xff, 0xb4, 0x86, 0xfa, 0x0c, 0x7e, 0x85, 0xa9, 0xca, 0x58, 0xc1, 0xea, 0xd5, 0xa6, 0x66, 0x79,
	0xc9, 0x2b, 0xad, 0x48, 0x6f, 0x8e, 0xa2, 0xf1, 0x82, 0x9c, 0x4e, 0x7a, 0xf3, 0x01, 0x7a, 0xe9,
	0x1a, 0x8d, 0x57, 0x06, 0x72, 0xe0, 0x99, 0x96, 0xc7, 0x90, 0xfe, 0x39, 0x88, 0x6b, 0xb4, 0x20,
	0x6b, 0x5e, 0x70, 0xcd, 0x8f, 0x20, 0x83, 0x73, 0x10, 0xd7, 0xf8, 0x87, 0xdc, 0x42, 0x3f, 0x2d,
	0x64, 0xaa, 0x48, 0x60, 0x9b, 0xb3, 0xd3, 0xe6, 0xb2, 0x90, 0x29, 0x75, 0x91, 0xf0, 0x01, 0x86,
	0x4d, 0x11, 0x4f, 0xa0, 0x2b, 0xd6, 0xfe, 0xf1, 0xbb, 0x62, 0x8d, 0x67, 0xd0, 0xdf, 0x88, 0x82,
	0x9b, 0x57, 0x47, 0xd1, 0x88, 0x3a, 0x13, 0x2e, 0xa1, 0x67, 0x00, 0x18, 0x43, 0xaf, 0x62, 0x25,
	0xf7, 0xfb, 0xb4, 0xda, 0x9c, 0x29, 0xf1, 0xcd, 0xed, 0x9a, 0x10, 0xb5, 0xda, 0x9c, 0x99, 0xa2,
	0x5d, 0xc4, 0x88, 0x5a, 0xbd, 0x7c, 0xf9, 0x7a, 0xce, 0x85, 0xde, 0xee, 0xd3, 0x38, 0x93, 0x65,
	0x52, 0x8a, 0xe2, 0xb0, 0x57, 0xf7, 0x42, 0x36, 0x4a, 0x69, 0x59, 0xb3, 0x9c, 0x27, 0xb9, 0x4c,
	0xec, 0x8d, 0x93, 0xf6, 0x07, 0xa4, 0x03, 0xfb, 0xf3, 0xf4, 0x1b, 0x00, 0x00, 0xff, 0xff, 0xfc,
	0xe7, 0x01, 0xb7, 0x8b, 0x02, 0x00, 0x00,
}

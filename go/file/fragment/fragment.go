package fragment

import "github.com/milvus-io/milvus-storage-format/proto/manifest_proto"

type FragmentType int32

const (
	kUnknown FragmentType = 0
	kData    FragmentType = 1
	kDelete  FragmentType = 2
)

type Fragment struct {
	fragmentId int64
	files      []string
}

type FragmentVector []Fragment

func NewFragment(fragmentId int64) *Fragment {
	return &Fragment{
		fragmentId: fragmentId,
	}
}

func (f *Fragment) AddFile(file string) {
	f.files = append(f.files, file)
}

func (f *Fragment) Files() []string {
	return f.files
}

func (f *Fragment) FragmentId() int64 {
	return f.fragmentId
}

func (f *Fragment) SetFragmentId(fragmentId int64) {
	f.fragmentId = fragmentId
}

func (f *Fragment) ToProtobuf() *manifest_proto.Fragment {
	fragment := &manifest_proto.Fragment{}
	fragment.Id = f.fragmentId
	for _, file := range f.files {
		fragment.Files = append(fragment.Files, file)
	}
	return fragment
}

func FromProtobuf(fragment *manifest_proto.Fragment) *Fragment {
	newFragment := NewFragment(fragment.Id)
	for _, file := range fragment.Files {
		newFragment.files = append(newFragment.files, file)
	}
	return newFragment
}

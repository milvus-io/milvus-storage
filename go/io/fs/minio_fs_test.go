package fs_test

import (
	"io"
	"testing"

	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/minio/minio-go/v7"
	"github.com/stretchr/testify/suite"
)

type MinioFsTestSuite struct {
	suite.Suite
	fs     fs.Fs
	client *minio.Client
}

func (suite *MinioFsTestSuite) SetupSuite() {
	fs, err := fs.BuildFileSystem("s3://minioadmin:minioadmin@localhost:9000/default")
	suite.NoError(err)
	suite.fs = fs
}

func (suite *MinioFsTestSuite) TestMinioOpenFile() {
	file, err := suite.fs.OpenFile("a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	file, err = suite.fs.OpenFile("a")
	suite.NoError(err)
	buf := make([]byte, 10)
	n, err = file.Read(buf)
	suite.Equal(io.EOF, err)
	suite.Equal(1, n)
	suite.ElementsMatch(buf[:n], []byte{1})
}

func (suite *MinioFsTestSuite) TestMinioRename() {
	file, err := suite.fs.OpenFile("a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	err = suite.fs.Rename("a", "b")
	suite.NoError(err)

	file, err = suite.fs.OpenFile("b")
	suite.NoError(err)
	buf := make([]byte, 10)
	n, err = file.Read(buf)
	suite.Equal(io.EOF, err)
	suite.Equal(1, n)
	suite.ElementsMatch(buf[:n], []byte{1})
}

func (suite *MinioFsTestSuite) TestMinioFsDeleteFile() {
	file, err := suite.fs.OpenFile("a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	err = suite.fs.DeleteFile("a")
	suite.NoError(err)

	exist, err := suite.fs.Exist("a")
	suite.NoError(err)
	suite.False(exist)
}

func (suite *MinioFsTestSuite) TestMinioFsList() {
	file, err := suite.fs.OpenFile("a/b")
	suite.NoError(err)
	_, err = file.Write([]byte{1})
	suite.NoError(err)
	suite.NoError(file.Close())
	file, err = suite.fs.OpenFile("a/b/c")
	suite.NoError(err)
	_, err = file.Write([]byte{1})
	suite.NoError(err)
	suite.NoError(file.Close())

	entries, err := suite.fs.List("a/")
	suite.NoError(err)
	suite.EqualValues([]fs.FileEntry{{Path: "a/b"}}, entries)
}

func (suite *MinioFsTestSuite) TestMinioFsReadFile() {
	file, err := suite.fs.OpenFile("a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	content, err := suite.fs.ReadFile("a")
	suite.NoError(err)
	suite.EqualValues([]byte{1}, content)
}

func (suite *MinioFsTestSuite) TestMinioFsExist() {
	exist, err := suite.fs.Exist("nonexist")
	suite.NoError(err)
	suite.False(exist)

	file, err := suite.fs.OpenFile("exist")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	exist, err = suite.fs.Exist("exist")
	suite.NoError(err)
	suite.True(exist)
}

func TestMinioFsSuite(t *testing.T) {
	suite.Run(t, &MinioFsTestSuite{})
}

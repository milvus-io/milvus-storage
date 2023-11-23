// Copyright 2023 Zilliz
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fs_test

import (
	"io"
	"testing"

	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/minio/minio-go/v7"
	"github.com/stretchr/testify/suite"
)

type MinioFsTestSuite struct {
	suite.Suite
	fs     fs.Fs
	client *minio.Client
}

func (suite *MinioFsTestSuite) SetupSuite() {
	fs, err := fs.BuildFileSystem("s3://minioadmin:minioadmin@default/path1?endpoint_override=localhost%3A9000")
	suite.NoError(err)
	suite.fs = fs
}

func (suite *MinioFsTestSuite) TestMinioOpenFile() {
	file, err := suite.fs.OpenFile("default/a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	file, err = suite.fs.OpenFile("default/a")
	suite.NoError(err)
	buf := make([]byte, 10)
	n, err = file.Read(buf)
	suite.Equal(io.EOF, err)
	suite.Equal(1, n)
	suite.ElementsMatch(buf[:n], []byte{1})

	suite.NoError(suite.fs.DeleteFile("default/a"))
}

func (suite *MinioFsTestSuite) TestMinioRename() {
	file, err := suite.fs.OpenFile("default/a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	err = suite.fs.Rename("default/a", "default/b")
	suite.NoError(err)

	file, err = suite.fs.OpenFile("default/b")
	suite.NoError(err)
	buf := make([]byte, 10)
	n, err = file.Read(buf)
	suite.Equal(io.EOF, err)
	suite.Equal(1, n)
	suite.ElementsMatch(buf[:n], []byte{1})
}

func (suite *MinioFsTestSuite) TestMinioFsDeleteFile() {
	file, err := suite.fs.OpenFile("default/a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	err = suite.fs.DeleteFile("default/a")
	suite.NoError(err)

	exist, err := suite.fs.Exist("default/a")
	suite.NoError(err)
	suite.False(exist)

}

func (suite *MinioFsTestSuite) TestMinioFsList() {
	file, err := suite.fs.OpenFile("default/a/b/c")
	suite.NoError(err)
	_, err = file.Write([]byte{1})
	suite.NoError(err)
	suite.NoError(file.Close())

	entries, err := suite.fs.List("default/a/")
	suite.NoError(err)
	suite.EqualValues([]fs.FileEntry{{Path: "default/a/b/c"}}, entries)

	suite.NoError(suite.fs.DeleteFile("default/a/b/c"))
}

func (suite *MinioFsTestSuite) TestMinioFsReadFile() {
	file, err := suite.fs.OpenFile("default/a")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	content, err := suite.fs.ReadFile("default/a")
	suite.NoError(err)
	suite.EqualValues([]byte{1}, content)
}

func (suite *MinioFsTestSuite) TestMinioFsExist() {
	exist, err := suite.fs.Exist("default/nonexist")
	suite.NoError(err)
	suite.False(exist)

	file, err := suite.fs.OpenFile("default/exist")
	suite.NoError(err)
	n, err := file.Write([]byte{1})
	suite.NoError(err)
	suite.Equal(1, n)
	suite.NoError(file.Close())

	exist, err = suite.fs.Exist("default/exist")
	suite.NoError(err)
	suite.True(exist)

	suite.NoError(suite.fs.DeleteFile("default/exist"))
}

func TestMinioFsSuite(t *testing.T) {
	suite.Run(t, &MinioFsTestSuite{})
}

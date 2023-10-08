package fs

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/milvus-io/milvus-storage/go/common/errors"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/io/fs/file"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"go.uber.org/zap"
)

type MinioFs struct {
	client     *minio.Client
	bucketName string
}

func ExtractFileName(path string) (string, error) {
	p := strings.Index(path, "/")
	if p == -1 {
		return "", errors.ErrInvalidPath
	}
	return path[p+1:], nil
}

func (fs *MinioFs) OpenFile(path string) (file.File, error) {
	fileName, err := ExtractFileName(path)
	if err != nil {
		return nil, err
	}
	return file.NewMinioFile(fs.client, fileName, fs.bucketName)
}

func (fs *MinioFs) Rename(src string, dst string) error {
	srcFileName, err := ExtractFileName(src)
	if err != nil {
		return err
	}
	dstFileName, err := ExtractFileName(dst)
	if err != nil {
		return err
	}
	_, err = fs.client.CopyObject(context.TODO(), minio.CopyDestOptions{Bucket: fs.bucketName, Object: dstFileName}, minio.CopySrcOptions{Bucket: fs.bucketName, Object: srcFileName})
	if err != nil {
		return err
	}
	err = fs.client.RemoveObject(context.TODO(), fs.bucketName, srcFileName, minio.RemoveObjectOptions{})
	if err != nil {
		log.Warn("failed to remove source object", log.String("source", srcFileName))
	}
	return nil
}

func (fs *MinioFs) DeleteFile(path string) error {
	fileName, err := ExtractFileName(path)
	if err != nil {
		return err
	}
	return fs.client.RemoveObject(context.TODO(), fs.bucketName, fileName, minio.RemoveObjectOptions{})
}

func (fs *MinioFs) CreateDir(path string) error {
	return nil
}

func (fs *MinioFs) List(path string) ([]FileEntry, error) {
	path, err := ExtractFileName(path)
	if err != nil {
		return nil, err
	}
	ret := make([]FileEntry, 0)
	for objInfo := range fs.client.ListObjects(context.TODO(), fs.bucketName, minio.ListObjectsOptions{Prefix: path, Recursive: true}) {
		if objInfo.Err != nil {
			log.Warn("list object error", zap.Error(objInfo.Err))
			return nil, objInfo.Err
		}
		ret = append(ret, FileEntry{Path: objInfo.Key})
	}
	return ret, nil
}

func (fs *MinioFs) ReadFile(path string) ([]byte, error) {
	path, err := ExtractFileName(path)
	if err != nil {
		return nil, err
	}
	obj, err := fs.client.GetObject(context.TODO(), fs.bucketName, path, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}

	stat, err := obj.Stat()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, stat.Size)
	n, err := obj.Read(buf)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if n != int(stat.Size) {
		return nil, fmt.Errorf("failed to read full file, expect: %d, actual: %d", stat.Size, n)
	}
	return buf, nil
}

func (fs *MinioFs) Exist(path string) (bool, error) {
	path, err := ExtractFileName(path)
	if err != nil {
		return false, err
	}
	_, err = fs.client.StatObject(context.TODO(), fs.bucketName, path, minio.StatObjectOptions{})
	if err != nil {
		resp := minio.ToErrorResponse(err)
		if resp.Code == "NoSuchKey" {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func (fs *MinioFs) Path() string {
	return fs.bucketName
}

// uri should be s3://accessKey:secretAceessKey@endpoint/bucket/
func NewMinioFs(uri *url.URL) (*MinioFs, error) {
	accessKey := uri.User.Username()
	secretAccessKey, set := uri.User.Password()
	if !set {
		log.Warn("secret access key not set")
	}
	cli, err := minio.New(uri.Host, &minio.Options{
		BucketLookup: minio.BucketLookupAuto,
		Creds:        credentials.NewStaticV4(accessKey, secretAccessKey, ""),
	})
	if err != nil {
		return nil, err
	}

	bucket := uri.Path
	if bucket[0] == '/' {
		bucket = bucket[1:]
	}
	exist, err := cli.BucketExists(context.TODO(), bucket)
	if err != nil {
		return nil, err
	}

	if !exist {
		if err = cli.MakeBucket(context.TODO(), bucket, minio.MakeBucketOptions{}); err != nil {
			return nil, err
		}
	}

	return &MinioFs{
		client:     cli,
		bucketName: bucket,
	}, nil
}

package fs

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/milvus-io/milvus-storage/go/common/constant"
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
	path       string
}

func (fs *MinioFs) MkdirAll(dir string, i int) error {
	//TODO implement me
	panic("implement me")
}

func ExtractFileName(path string) (string, error) {
	p := strings.Index(path, "/")
	if p == -1 {
		return "", errors.ErrInvalidPath
	}
	return path[p+1:], nil
}

func (fs *MinioFs) OpenFile(path string) (file.File, error) {
	return file.NewMinioFile(fs.client, path, fs.bucketName)
}

func (fs *MinioFs) Rename(src string, dst string) error {
	_, err := fs.client.CopyObject(context.TODO(), minio.CopyDestOptions{Bucket: fs.bucketName, Object: dst}, minio.CopySrcOptions{Bucket: fs.bucketName, Object: src})
	if err != nil {
		return err
	}
	err = fs.client.RemoveObject(context.TODO(), fs.bucketName, src, minio.RemoveObjectOptions{})
	if err != nil {
		log.Warn("failed to remove source object", log.String("source", src))
	}
	return nil
}

func (fs *MinioFs) DeleteFile(path string) error {
	return fs.client.RemoveObject(context.TODO(), fs.bucketName, path, minio.RemoveObjectOptions{})
}

func (fs *MinioFs) CreateDir(path string) error {
	return nil
}

func (fs *MinioFs) List(path string) ([]FileEntry, error) {
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
	_, err := fs.client.StatObject(context.TODO(), fs.bucketName, path, minio.StatObjectOptions{})
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
	return fs.path
}

// uri should be s3://username:password@bucket/path?endpoint_override=localhost%3A9000
func NewMinioFs(uri *url.URL) (*MinioFs, error) {
	accessKey := uri.User.Username()
	secretAccessKey, set := uri.User.Password()
	if !set {
		log.Warn("secret access key not set")
	}

	endpoints, ok := uri.Query()[constant.EndpointOverride]
	if !ok || len(endpoints) == 0 {
		return nil, errors.ErrNoEndpoint
	}

	cli, err := minio.New(endpoints[0], &minio.Options{
		BucketLookup: minio.BucketLookupAuto,
		Creds:        credentials.NewStaticV4(accessKey, secretAccessKey, ""),
	})
	if err != nil {
		return nil, err
	}

	bucket := uri.Host
	path := uri.Path

	log.Info("minio fs infos", zap.String("endpoint", endpoints[0]), zap.String("bucket", bucket), zap.String("path", path))

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
		path:       path,
	}, nil
}

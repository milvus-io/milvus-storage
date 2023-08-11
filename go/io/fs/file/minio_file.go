package file

import (
	"bytes"
	"context"

	"github.com/minio/minio-go/v7"
)

var _ File = (*MinioFile)(nil)

type MinioFile struct {
	*minio.Object
	writer     *MemoryFile
	client     *minio.Client
	fileName   string
	bucketName string
}

func (f *MinioFile) Write(b []byte) (int, error) {
	return f.writer.Write(b)
}

func (f *MinioFile) Close() error {
	if f.writer == nil {
		return nil
	}
	_, err := f.client.PutObject(context.TODO(), f.bucketName, f.fileName, bytes.NewReader(f.writer.b), int64(len(f.writer.b)), minio.PutObjectOptions{})
	return err
}

func NewMinioFile(client *minio.Client, fileName string, bucketName string) (*MinioFile, error) {
	_, err := client.StatObject(context.TODO(), bucketName, fileName, minio.StatObjectOptions{})
	if err != nil {
		eresp := minio.ToErrorResponse(err)
		if eresp.Code != "NoSuchKey" {
			return nil, err
		}
		return &MinioFile{
			writer:     NewMemoryFile(nil),
			client:     client,
			fileName:   fileName,
			bucketName: bucketName,
		}, nil
	}

	object, err := client.GetObject(context.TODO(), bucketName, fileName, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}

	return &MinioFile{
		Object:     object,
		client:     client,
		fileName:   fileName,
		bucketName: bucketName,
	}, nil
}

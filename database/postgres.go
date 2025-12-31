package database

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	pgxuuid "github.com/jackc/pgx-gofrs-uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PgxConnIface interface {
	Begin(context.Context) (pgx.Tx, error)
	BeginTx(ctx context.Context, txOptions pgx.TxOptions) (pgx.Tx, error)
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
	Ping(context.Context) error
	Prepare(context.Context, string, string) (*pgconn.StatementDescription, error)
	Close(context.Context) error
	CopyFrom(ctx context.Context, tableName pgx.Identifier, columnNames []string, rowSrc pgx.CopyFromSource) (int64, error)
}
type PgxIface interface {
	Acquire(ctx context.Context) (c *pgxpool.Conn, err error)
	Begin(context.Context) (pgx.Tx, error)
	BeginTx(ctx context.Context, txOptions pgx.TxOptions) (pgx.Tx, error)
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
	Ping(context.Context) error
	Close()
}

func Config(dbUrl string) *pgxpool.Config {
	dbConfig, err := pgxpool.ParseConfig(dbUrl)
	if err != nil {
		slog.Error("Failed to create a config", slog.String("error", err.Error()))
		panic(err)
	}
	dbConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		pgxuuid.Register(conn.TypeMap())
		return nil
	}

	dbConfig.MaxConns = int32(maxOpenConns)
	dbConfig.MaxConnLifetime = connMaxLifetime
	dbConfig.MaxConnIdleTime = connMaxIdleTime
	dbConfig.HealthCheckPeriod = healthCheckPeriod
	return dbConfig
}

func NewPostgresDB(username, password, host, name, port, schema string) PgxIface {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dbUrl := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=disable&search_path=%s", username, password, host, port, name, schema)
	connPool, err := pgxpool.NewWithConfig(ctx, Config(dbUrl))
	if err != nil {
		slog.Error("error while creating connection to the database!!", slog.String("error", err.Error()))
		panic(err)
	}

	connection, err := connPool.Acquire(ctx)
	if err != nil {
		slog.Error("error while acquiring connection from the database pool!!", slog.String("error", err.Error()))
		panic(err)
	}
	defer connection.Release()

	err = connection.Ping(ctx)
	if err != nil {
		slog.Error("could not ping database", slog.String("error", err.Error()))
		panic(err)
	}

	return connPool
}

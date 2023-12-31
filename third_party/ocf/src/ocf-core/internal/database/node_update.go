// Code generated by ent, DO NOT EDIT.

package database

import (
	"context"
	"errors"
	"fmt"
	"ocfcore/internal/database/node"
	"ocfcore/internal/database/predicate"

	"entgo.io/ent/dialect/sql"
	"entgo.io/ent/dialect/sql/sqlgraph"
	"entgo.io/ent/schema/field"
)

// NodeUpdate is the builder for updating Node entities.
type NodeUpdate struct {
	config
	hooks    []Hook
	mutation *NodeMutation
}

// Where appends a list predicates to the NodeUpdate builder.
func (nu *NodeUpdate) Where(ps ...predicate.Node) *NodeUpdate {
	nu.mutation.Where(ps...)
	return nu
}

// SetPeerId sets the "peerId" field.
func (nu *NodeUpdate) SetPeerId(s string) *NodeUpdate {
	nu.mutation.SetPeerId(s)
	return nu
}

// SetNillablePeerId sets the "peerId" field if the given value is not nil.
func (nu *NodeUpdate) SetNillablePeerId(s *string) *NodeUpdate {
	if s != nil {
		nu.SetPeerId(*s)
	}
	return nu
}

// SetStatus sets the "status" field.
func (nu *NodeUpdate) SetStatus(s string) *NodeUpdate {
	nu.mutation.SetStatus(s)
	return nu
}

// SetNillableStatus sets the "status" field if the given value is not nil.
func (nu *NodeUpdate) SetNillableStatus(s *string) *NodeUpdate {
	if s != nil {
		nu.SetStatus(*s)
	}
	return nu
}

// Mutation returns the NodeMutation object of the builder.
func (nu *NodeUpdate) Mutation() *NodeMutation {
	return nu.mutation
}

// Save executes the query and returns the number of nodes affected by the update operation.
func (nu *NodeUpdate) Save(ctx context.Context) (int, error) {
	var (
		err      error
		affected int
	)
	if len(nu.hooks) == 0 {
		affected, err = nu.sqlSave(ctx)
	} else {
		var mut Mutator = MutateFunc(func(ctx context.Context, m Mutation) (Value, error) {
			mutation, ok := m.(*NodeMutation)
			if !ok {
				return nil, fmt.Errorf("unexpected mutation type %T", m)
			}
			nu.mutation = mutation
			affected, err = nu.sqlSave(ctx)
			mutation.done = true
			return affected, err
		})
		for i := len(nu.hooks) - 1; i >= 0; i-- {
			if nu.hooks[i] == nil {
				return 0, fmt.Errorf("database: uninitialized hook (forgotten import database/runtime?)")
			}
			mut = nu.hooks[i](mut)
		}
		if _, err := mut.Mutate(ctx, nu.mutation); err != nil {
			return 0, err
		}
	}
	return affected, err
}

// SaveX is like Save, but panics if an error occurs.
func (nu *NodeUpdate) SaveX(ctx context.Context) int {
	affected, err := nu.Save(ctx)
	if err != nil {
		panic(err)
	}
	return affected
}

// Exec executes the query.
func (nu *NodeUpdate) Exec(ctx context.Context) error {
	_, err := nu.Save(ctx)
	return err
}

// ExecX is like Exec, but panics if an error occurs.
func (nu *NodeUpdate) ExecX(ctx context.Context) {
	if err := nu.Exec(ctx); err != nil {
		panic(err)
	}
}

func (nu *NodeUpdate) sqlSave(ctx context.Context) (n int, err error) {
	_spec := &sqlgraph.UpdateSpec{
		Node: &sqlgraph.NodeSpec{
			Table:   node.Table,
			Columns: node.Columns,
			ID: &sqlgraph.FieldSpec{
				Type:   field.TypeInt,
				Column: node.FieldID,
			},
		},
	}
	if ps := nu.mutation.predicates; len(ps) > 0 {
		_spec.Predicate = func(selector *sql.Selector) {
			for i := range ps {
				ps[i](selector)
			}
		}
	}
	if value, ok := nu.mutation.PeerId(); ok {
		_spec.SetField(node.FieldPeerId, field.TypeString, value)
	}
	if value, ok := nu.mutation.Status(); ok {
		_spec.SetField(node.FieldStatus, field.TypeString, value)
	}
	if n, err = sqlgraph.UpdateNodes(ctx, nu.driver, _spec); err != nil {
		if _, ok := err.(*sqlgraph.NotFoundError); ok {
			err = &NotFoundError{node.Label}
		} else if sqlgraph.IsConstraintError(err) {
			err = &ConstraintError{msg: err.Error(), wrap: err}
		}
		return 0, err
	}
	return n, nil
}

// NodeUpdateOne is the builder for updating a single Node entity.
type NodeUpdateOne struct {
	config
	fields   []string
	hooks    []Hook
	mutation *NodeMutation
}

// SetPeerId sets the "peerId" field.
func (nuo *NodeUpdateOne) SetPeerId(s string) *NodeUpdateOne {
	nuo.mutation.SetPeerId(s)
	return nuo
}

// SetNillablePeerId sets the "peerId" field if the given value is not nil.
func (nuo *NodeUpdateOne) SetNillablePeerId(s *string) *NodeUpdateOne {
	if s != nil {
		nuo.SetPeerId(*s)
	}
	return nuo
}

// SetStatus sets the "status" field.
func (nuo *NodeUpdateOne) SetStatus(s string) *NodeUpdateOne {
	nuo.mutation.SetStatus(s)
	return nuo
}

// SetNillableStatus sets the "status" field if the given value is not nil.
func (nuo *NodeUpdateOne) SetNillableStatus(s *string) *NodeUpdateOne {
	if s != nil {
		nuo.SetStatus(*s)
	}
	return nuo
}

// Mutation returns the NodeMutation object of the builder.
func (nuo *NodeUpdateOne) Mutation() *NodeMutation {
	return nuo.mutation
}

// Select allows selecting one or more fields (columns) of the returned entity.
// The default is selecting all fields defined in the entity schema.
func (nuo *NodeUpdateOne) Select(field string, fields ...string) *NodeUpdateOne {
	nuo.fields = append([]string{field}, fields...)
	return nuo
}

// Save executes the query and returns the updated Node entity.
func (nuo *NodeUpdateOne) Save(ctx context.Context) (*Node, error) {
	var (
		err  error
		node *Node
	)
	if len(nuo.hooks) == 0 {
		node, err = nuo.sqlSave(ctx)
	} else {
		var mut Mutator = MutateFunc(func(ctx context.Context, m Mutation) (Value, error) {
			mutation, ok := m.(*NodeMutation)
			if !ok {
				return nil, fmt.Errorf("unexpected mutation type %T", m)
			}
			nuo.mutation = mutation
			node, err = nuo.sqlSave(ctx)
			mutation.done = true
			return node, err
		})
		for i := len(nuo.hooks) - 1; i >= 0; i-- {
			if nuo.hooks[i] == nil {
				return nil, fmt.Errorf("database: uninitialized hook (forgotten import database/runtime?)")
			}
			mut = nuo.hooks[i](mut)
		}
		v, err := mut.Mutate(ctx, nuo.mutation)
		if err != nil {
			return nil, err
		}
		nv, ok := v.(*Node)
		if !ok {
			return nil, fmt.Errorf("unexpected node type %T returned from NodeMutation", v)
		}
		node = nv
	}
	return node, err
}

// SaveX is like Save, but panics if an error occurs.
func (nuo *NodeUpdateOne) SaveX(ctx context.Context) *Node {
	node, err := nuo.Save(ctx)
	if err != nil {
		panic(err)
	}
	return node
}

// Exec executes the query on the entity.
func (nuo *NodeUpdateOne) Exec(ctx context.Context) error {
	_, err := nuo.Save(ctx)
	return err
}

// ExecX is like Exec, but panics if an error occurs.
func (nuo *NodeUpdateOne) ExecX(ctx context.Context) {
	if err := nuo.Exec(ctx); err != nil {
		panic(err)
	}
}

func (nuo *NodeUpdateOne) sqlSave(ctx context.Context) (_node *Node, err error) {
	_spec := &sqlgraph.UpdateSpec{
		Node: &sqlgraph.NodeSpec{
			Table:   node.Table,
			Columns: node.Columns,
			ID: &sqlgraph.FieldSpec{
				Type:   field.TypeInt,
				Column: node.FieldID,
			},
		},
	}
	id, ok := nuo.mutation.ID()
	if !ok {
		return nil, &ValidationError{Name: "id", err: errors.New(`database: missing "Node.id" for update`)}
	}
	_spec.Node.ID.Value = id
	if fields := nuo.fields; len(fields) > 0 {
		_spec.Node.Columns = make([]string, 0, len(fields))
		_spec.Node.Columns = append(_spec.Node.Columns, node.FieldID)
		for _, f := range fields {
			if !node.ValidColumn(f) {
				return nil, &ValidationError{Name: f, err: fmt.Errorf("database: invalid field %q for query", f)}
			}
			if f != node.FieldID {
				_spec.Node.Columns = append(_spec.Node.Columns, f)
			}
		}
	}
	if ps := nuo.mutation.predicates; len(ps) > 0 {
		_spec.Predicate = func(selector *sql.Selector) {
			for i := range ps {
				ps[i](selector)
			}
		}
	}
	if value, ok := nuo.mutation.PeerId(); ok {
		_spec.SetField(node.FieldPeerId, field.TypeString, value)
	}
	if value, ok := nuo.mutation.Status(); ok {
		_spec.SetField(node.FieldStatus, field.TypeString, value)
	}
	_node = &Node{config: nuo.config}
	_spec.Assign = _node.assignValues
	_spec.ScanValues = _node.scanValues
	if err = sqlgraph.UpdateNode(ctx, nuo.driver, _spec); err != nil {
		if _, ok := err.(*sqlgraph.NotFoundError); ok {
			err = &NotFoundError{node.Label}
		} else if sqlgraph.IsConstraintError(err) {
			err = &ConstraintError{msg: err.Error(), wrap: err}
		}
		return nil, err
	}
	return _node, nil
}

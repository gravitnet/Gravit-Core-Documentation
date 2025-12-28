# L2 - Continuum Data Models

## 1. Semantic Unit (S-unit)

```
SUnit {
    sunit_id: Hash
    source_event: EventID
    meaning_vector: Vector[float]
    context_signature: ContextID
}
```

## 2. Semantic Relation

```
SRelation {
    a: SUnitID
    b: SUnitID
    relation_type: Enum(RelatesTo, Causes, Contradicts, Mirrors, Extends)
    strength: float
}
```

## 3. Context Model

```
Context {
    context_id: Hash
    participants: [EntityID]
    semantic_boundaries: Map(Key, Value)
    trust_baseline: Float
}
```

## 4. Continuum Flow Model

```
Flow {
    flow_id: Hash
    input_sunits: [SUnitID]
    transformation_ops: [TransformationOp]
    output_sunits: [SUnitID]
}
```

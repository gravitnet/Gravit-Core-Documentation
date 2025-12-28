# L1 - Open Network Data Models

## 1. Entity Primitives

```
Entity {
    entity_id: Hash
    type: EntityType
    created_at: Timestamp
    signatures: [Signature]
}
```

## 2. Event Model

```
Event {
    event_id: Hash
    entity_id: Hash
    event_type: EventType
    payload: Bytes
    integrity_proof: MerklePath
}
```

## 3. Observation Model

```
Observation {
    observer_id: Hash
    event_id: Hash
    timestamp: Timestamp
    confidence: Float
}
```

## 4. Intent Model

```
Intent {
    intent_id: Hash
    source_entity: Hash
    target_entity: Hash
    expected_outcome: OutcomeDefinition
}
```

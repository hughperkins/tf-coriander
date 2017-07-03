# Event Manager

This document is general to Tensorflow, not specific to Tensorflow-cl. It is my notes on how the event manager works, to try to solve a horrible bug related to it https://github.com/hughperkins/tensorflow-cl/issues/34

Approximate diagram of how the event manager looks:

<img src="img/eventmgr.png" />

## Life cycle of a function/event

### Part 1: `ThenExecute`

- something calls `ThenExecute`, passing in a function
- the function gets added to an `InUse` struct, and the `InUse` struct gets added to `used_events_` vector
  - `used_events_` holds `InUse` structs, despite sounding 'event'y
  - its an object property, fo the EventMgr class
- before being added to `used_events_`, an Event object is either pulled from the `free_events_` vector, or created from scratch
  - this gets assign to the InUse struct 'event' property

So, at this point we have:

- in the `ThenExecute` method:
  - a vector `to_free`, currently empty
- in the EventMgr object:
  - a vector `used_events_`, containing the `InUse` struct, that contains the function

In passing, note that the new event object was passed into `Stream->RecordEvent(.)`, just after being created

### Part 2: polling

- `PollLoop` (or possibly the initial `ThenExecute`) calls `PollEvents`, passing in a currently-empty vector `to_free`
- `PollEvents` calls `PollForStatus` on each event object, on each of the `InUse` objects in the `used_events_` property of EventMgr
- if any of the events are completed:
  - the `InUse` object is added to the `to_free` vector, from the calling `PollLoop` method
  - the `Event` itself is added to the property `free_events_`
  - the `event` property of the `InUse` object is set to `nullptr`
- any `InUse` objects at the front of the `used_events_` vector are removed from it, considered 'finished'
  - any with some in front of them are not removed yet
  - but will probably be removed later. They're still basically no longer used though
- note that at this point, the function, in the `func` property of the `InUse` object has not been called yet
  - however, the function is stored in the `to_free` vector, in the calling `PollLoop` method
  - so, it's important that it's stored there by-value

### Part 3: execution and free

- the `PollLoop` (or possibly the initial `ThenExecute`) method calls `FreeMemory`, passing in the list of completed `InUse` structs, from calling `PollEvents` earlier
- for each of the `InUse` structs in `to_free`, `FreeMemory` does:

(at this point, I had figured out how to fix the problem :-) )

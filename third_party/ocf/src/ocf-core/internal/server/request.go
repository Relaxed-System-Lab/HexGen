package server

import (
	"encoding/json"
	"ocfcore/internal/common/requests"
	"ocfcore/internal/common/structs"
	"ocfcore/internal/protocol/p2p"
	"ocfcore/internal/server/queue"

	"github.com/gin-gonic/gin"
)

var rrIndex int

type InferenceResponse struct {
	Msg  string `json:"msg"`
	Data string `json:"data"`
}

func InferenceRequest(c *gin.Context) {	
	var request structs.InferenceStruct
	err := c.BindJSON(&request)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	jsonRequest, err := json.Marshal(request)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	topic := "inference:" + request.UniqueModelName
	msg, err := queue.Publish(topic, jsonRequest)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	// wait until the inference is done
	c.JSON(200, gin.H{"message": "ok", "data": string(msg.Data)})
}

// AutoInferenceRequest is a function that handles the inference request, but dispatches it to the correct worker
// todo: we should have the ability to "cleverly" dispatch the inference request to the "fastest" worker
func AutoInferenceRequest(c *gin.Context) {
	var request structs.InferenceStruct
	err := c.BindJSON(&request)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	// find workers
	table := p2p.GetNodeTable()
	topic := "inference:" + request.UniqueModelName
	providers := table.FindProviders(topic)
	if len(providers) == 0 {
		c.JSON(500, gin.H{"error": "no worker available"})
		return
	}

	// pick a worker by round robin method
	var scapegoat p2p.Peer
	if rrIndex < len(providers) {
		scapegoat = providers[rrIndex]
	} else {
		// in case someone leave 
		rrIndex = len(providers) - 1
		scapegoat = providers[rrIndex]
	}
	rrIndex = (rrIndex + 1) % len(providers)
	// now forward request to scapegoat
	res, err := requests.ForwardInferenceRequest(scapegoat.PeerID, request)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
	}
	var response InferenceResponse
	err = json.Unmarshal([]byte(res), &response)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
	}
	c.JSON(200, gin.H{"message": "ok", "data": response.Data})
}
